"""Watches Dream-1 work and judges, from the overhead camera, when its task is done or stuck.

    uv run --group monitor task-monitor             # pauses Dream-1 on a verdict and tells the voice agent
    uv run --group monitor task-monitor --dry-run   # reports only, never pauses: calibrate with this first

A Portal observer, like the recorder: it never sends actions. While `uv run dream-1-tasks` drives, it keeps the
camera's frames since Dream-1 took control, and every few seconds asks LeRobot's TOPReward
(huggingface.co/docs/lerobot/topreward) how a vision-language model answers "this video shows a robot completing
<task>: True or False?". The share of True is the progress estimate. An estimate that stays high for a while means
done; an estimate that stops rising, or arms that stop moving, means stuck. Either way the monitor clears the robot's
control gate, as a pause does, so the arms hold. Only "continue" (or `c`) gives Dream-1 control again.

Every few seconds, and at once on a verdict, it sends the voice agent its report over the `monitor.report` RPC
(src/utils/common.py), so the agent can say how the task is going and announce the verdict.
See tutorial/12-task-monitor.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from livekit.portal import Observer, ObserverConfig, frame_bytes_to_numpy_rgb
from utils.common import (portal_config, DREAM_STATUS_RPC, DREAM_TASKS_IDENTITY, MONITOR_IDENTITY, MONITOR_REPORT_RPC,
                          VOICE_AGENT_IDENTITY, HardwareThread, env_float, env_str, load_env, mint_token, pace,
                          required_env)
from utils.metal import GRIPPER_MAX_MM, SIDES, arm_fields

WHO = "task-monitor"
TICK_HZ = 4  # the monitor's own loop: frames, arms, scores, reports
CLIP_HZ = 2.0  # frames kept per second at the start of a run; halved each time the clip fills
CLIP_CAP = 64  # frames held at most
FRAME_WIDTH = 448  # frames shrink to this width as they arrive; the model's processor sizes them from there
VLM_FRAMES = 16  # frames per score, spread from the start of the run to now (TOPReward's own default)
MIN_FRAMES = 4  # a run's first score waits for this many (about two seconds)
TREND_S = 10.0  # the trend compares the estimate with this long ago
MOVED = 0.02  # a joint (rad) or a grip (share of its range) changing more than this counts as motion
REPORT_S = 5.0  # report at least this often, so the voice agent knows the monitor runs
RETRY_S = 3.0  # wait between attempts to read Dream-1's task


@dataclass(frozen=True)
class Limits:
    """When a run is done or stuck. Each has an environment variable; see tutorial/12-task-monitor.md."""

    success: float = 0.85  # MONITOR_SUCCESS: every score at or above this ...
    hold_s: float = 10.0  # MONITOR_SUCCESS_HOLD_S: ... for this long: done (Dream-1 keeps going meanwhile)
    stuck_s: float = 45.0  # MONITOR_STUCK_S: the estimate has not risen by `gain` in this long: stuck
    gain: float = 0.05  # MONITOR_GAIN
    still_s: float = 15.0  # MONITOR_STILL_S: the arms have not moved in this long: stuck
    every_s: float = 2.0  # MONITOR_EVERY_S: seconds between scores, if the model keeps up

    @classmethod
    def from_env(cls) -> Limits:
        return cls(success=env_float("MONITOR_SUCCESS", cls.success),
                   hold_s=env_float("MONITOR_SUCCESS_HOLD_S", cls.hold_s),
                   stuck_s=env_float("MONITOR_STUCK_S", cls.stuck_s), gain=env_float("MONITOR_GAIN", cls.gain),
                   still_s=env_float("MONITOR_STILL_S", cls.still_s), every_s=env_float("MONITOR_EVERY_S", cls.every_s))


def shrink(frame) -> np.ndarray:
    """A received frame as a FRAME_WIDTH-wide RGB array, a copy the clip can keep."""
    rgb = frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height)
    height = max(2, round(frame.height * FRAME_WIDTH / frame.width))
    return cv2.resize(rgb, (FRAME_WIDTH, height), interpolation=cv2.INTER_AREA)


def pose(state: dict) -> np.ndarray:
    """Both arms' joints in radians and grips as a share of their range, for telling whether they move."""
    values = []
    for side in SIDES:
        *joints, gripper = arm_fields(side)
        values += [state[name] for name in joints] + [state[gripper] / GRIPPER_MAX_MM]
    return np.array(values, dtype=np.float64)


class Clip:
    """The camera's frames since the run started, evenly spaced, in bounded memory.

    It keeps a frame every `1 / hz` seconds. Full at `cap`, it drops every other frame and halves the rate, so it
    always spans the whole run however long that is.
    """

    def __init__(self, hz: float = CLIP_HZ, cap: int = CLIP_CAP) -> None:
        self.interval, self.cap = 1.0 / hz, cap
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []

    def __len__(self) -> int:
        return len(self.frames)

    def wants(self, t: float) -> bool:
        return not self.times or t - self.times[-1] >= self.interval

    def add(self, t: float, frame: np.ndarray) -> None:
        self.frames.append(frame)
        self.times.append(t)
        if len(self.frames) >= self.cap:
            self.frames, self.times = self.frames[::2], self.times[::2]
            self.interval *= 2

    def sample(self, n: int) -> tuple[np.ndarray, float]:
        """Up to `n` frames spread over the clip, its first and newest included, and their rate in frames per
        second: [frames, height, width, 3] uint8."""
        picks = np.unique(np.linspace(0, len(self.frames) - 1, min(n, len(self.frames))).round().astype(int))
        span = self.times[picks[-1]] - self.times[picks[0]]
        return np.stack([self.frames[i] for i in picks]), (len(picks) - 1) / span if span > 0 else 1.0


class Run:
    """One stretch of Dream-1 driving on one task: its frames and scores, and the verdict on it.

    `progress` is the newest score, the number the log prints and the voice agent says. Whether the estimate still
    rises, and its trend, use the median of the last three scores, so one outlier moves neither.
    """

    def __init__(self, number: int, operator: str, task: str, started: float, limits: Limits) -> None:
        self.number, self.operator, self.task, self.started, self.limits = number, operator, task, started, limits
        self.clip = Clip()
        self.scores: list[float] = []
        self.levels: list[tuple[float, float]] = []  # (when, median of the last three scores)
        self.mark, self.rose_at = -math.inf, started  # the level it last rose to by `gain`, and when
        self.high_since: float | None = None  # when the scores reached `success` and have stayed there since
        self.pose, self.moved_at = None, started
        self.verdict: str | None = None  # "success" or "stuck", decided once
        self.reason = ""
        self.released = False  # the monitor took control from Dream-1
        self.ended: float | None = None  # when the run stopped being watched

    @property
    def progress(self) -> float | None:
        return self.scores[-1] if self.scores else None

    def done_for(self, t: float) -> float | None:
        """Seconds the scores have stayed at or above `success`, or None."""
        return None if self.high_since is None else t - self.high_since

    def add_score(self, t: float, p: float) -> str | None:
        """A score of the clip up to `t`; returns a new verdict, if this decides one."""
        self.scores.append(p)
        level = float(np.median(self.scores[-3:]))
        self.levels.append((t, level))
        if level >= self.mark + self.limits.gain:
            self.mark, self.rose_at = level, t
        if p < self.limits.success:
            self.high_since = None  # a dip starts the wait again: the task may have come undone
        elif self.high_since is None:
            self.high_since = t
        if (held := self.done_for(t)) is not None:
            if held >= self.limits.hold_s:
                return self._decide("success", f"an estimated {p:.0%} done, and above {self.limits.success:.0%} "
                                               f"for {held:.0f} seconds")
        elif t - self.rose_at >= self.limits.stuck_s:
            return self._decide("stuck", f"no progress in {t - self.rose_at:.0f} seconds")
        return None

    def add_pose(self, t: float, values: np.ndarray) -> str | None:
        """Where the arms are at `t`; returns a new verdict, if they have been still too long. Arms that stop while
        the estimate looks done are finishing, not stuck."""
        if self.pose is None or np.abs(values - self.pose).max() > MOVED:
            self.pose, self.moved_at = values, t
        elif t - self.moved_at >= self.limits.still_s and self.high_since is None:
            return self._decide("stuck", f"the arms have not moved in {t - self.moved_at:.0f} seconds")
        return None

    def _decide(self, verdict: str, reason: str) -> str | None:
        if self.verdict is not None:
            return None
        self.verdict, self.reason = verdict, reason
        return verdict

    def trend(self) -> str | None:
        """Rising, flat, or falling: the estimate now against TREND_S ago."""
        if len(self.levels) < 2:
            return None
        now, level = self.levels[-1]
        then = next((old for t, old in reversed(self.levels) if t <= now - TREND_S), self.levels[0][1])
        return "rising" if level - then > self.limits.gain / 2 else "falling" if then - level > self.limits.gain / 2 \
            else "flat"

    def report(self, now: float) -> dict:
        progress, held = self.progress, self.done_for(now)
        return {"state": self.verdict or ("watching" if self.ended is None else "stopped"), "run": self.number,
                "task": self.task, "operator": self.operator,
                "elapsed_s": round((self.ended or now) - self.started, 1),
                "progress": None if progress is None else round(progress, 3), "trend": self.trend(),
                "done_for_s": None if held is None or self.verdict else round(held, 1),
                "hold_s": self.limits.hold_s, "reason": self.reason, "released": self.released}


class Scorer:
    """TOPReward's question, asked of Qwen3-VL through LeRobot's encoder: does this video complete the task?

    LeRobot's reward model returns log P("True"), whose scale depends on the model and the scene. This reads the
    same forward pass for "False" too, and returns P(True) / (P(True) + P(False)), a 0-1 estimate, with the raw
    log P("True") alongside.
    """

    def __init__(self, vlm: str, device: str | None) -> None:
        import torch  # the monitor group's dependencies load only here
        from lerobot.lerobot_types import TransitionKey
        from lerobot.rewards.topreward import TOPRewardConfig, TOPRewardModel
        from lerobot.rewards.topreward.processor_topreward import (TOPREWARD_FEATURE_PREFIX,
                                                                   TOPREWARD_VLM_INPUT_KEYS,
                                                                   TOPRewardEncoderProcessorStep)
        from lerobot.utils.device_utils import auto_select_torch_device
        from transformers import AutoTokenizer

        self.torch, self.keys = torch, TransitionKey
        self.inputs = [TOPREWARD_FEATURE_PREFIX + key for key in TOPREWARD_VLM_INPUT_KEYS]
        device = device or auto_select_torch_device().type  # CUDA, else Apple's MPS, else the CPU
        config = TOPRewardConfig(vlm_name=vlm, device=device)
        self.device = config.device
        self.model = TOPRewardModel(config).to(self.device).eval()
        # Frames are already chosen (Clip.sample); the encoder writes the prompt and the video tokens.
        self.encoder = TOPRewardEncoderProcessorStep(vlm_name=vlm, image_key=config.image_key, max_frames=None)
        tokenizer = AutoTokenizer.from_pretrained(vlm)

        def answer(word: str) -> int:
            ids = tokenizer(f"The answer is: {word}", add_special_tokens=False).input_ids
            if tokenizer.decode(ids[-1:]).strip() != word:
                raise RuntimeError(f"{vlm} splits {word!r} into several tokens; TOPReward needs a Qwen3-VL model")
            return ids[-1]

        self.true, self.false = answer("True"), answer("False")

    def score(self, frames: np.ndarray, fps: float, task: str) -> tuple[float, float]:
        """(P(True) / (P(True) + P(False)), log P(True)) for `frames` [T, H, W, 3] uint8 played at `fps`."""
        torch = self.torch
        self.encoder.fps = fps  # timestamps for the video tokens
        transition = {self.keys.OBSERVATION: {self.encoder.image_key: torch.from_numpy(frames).unsqueeze(0)},
                      self.keys.COMPLEMENTARY_DATA: {"task": task}}
        encoded = self.encoder(transition)[self.keys.OBSERVATION]
        inputs = {key.rsplit(".", 1)[-1]: encoded[key].to(self.device) for key in self.inputs if key in encoded}
        with torch.inference_mode():
            logits = self.model.model(**inputs, logits_to_keep=2).logits[0, -2].float()  # predicts the answer
        log_p = torch.log_softmax(logits, dim=-1)
        true, false = float(log_p[self.true]), float(log_p[self.false])
        return 1.0 / (1.0 + math.exp(false - true)), true


class Monitor:
    """The monitor's loop body, one `tick` at a time: follows who drives, scores runs, decides, and reports.

    `scorer` is anything with `score(frames, fps, task)`; tests pass a stub.
    """

    def __init__(self, obs, scorer, worker: HardwareThread, limits: Limits, camera: str, dry_run: bool) -> None:
        self.obs, self.scorer, self.worker = obs, scorer, worker
        self.limits, self.camera, self.dry_run = limits, camera, dry_run
        self.run: Run | None = None
        self.runs = 0
        self.scoring = None  # (run, asked at, future) of the score in flight
        self.scored_at = -math.inf
        self.retry_at = 0.0
        self.reported_at = -math.inf
        self.pending: dict | None = None  # the newest report not yet sent
        self.sending: asyncio.Future | None = None
        self.heard: bool | None = None  # whether the voice agent took the last report

    def watching(self) -> bool:
        return self.run is not None and self.run.ended is None

    async def tick(self, now: float) -> None:
        driver, run = self.obs.active_operator(), self.run
        if self.watching() and driver != run.operator:
            run.ended = now  # paused, handed to the teleoperator, or parked
            print(f"[{WHO}] run {run.number}: {driver or 'nobody'} has control; stopped watching")
            self.report(now)
        if driver == DREAM_TASKS_IDENTITY and not self.watching() and now >= self.retry_at:
            await self.start(driver, now)
        if self.watching():
            await self.watch(self.run, now)
        if now - self.reported_at >= REPORT_S:
            self.report(now)

    async def start(self, driver: str, now: float) -> None:
        """A new run: Dream-1 just took control. Its task comes from Dream-1 itself."""
        try:
            status = json.loads(await self.obs.perform_rpc(DREAM_STATUS_RPC, "{}", destination=driver,
                                                           response_timeout_ms=3000))
        except Exception as exc:
            self.retry_at = now + RETRY_S
            print(f"[{WHO}] cannot read Dream-1's task ({exc}); trying again")
            return
        if not status.get("task"):  # handed control with no task, Dream-1 does not move: nothing to watch
            self.retry_at = now + RETRY_S
            return
        self.runs += 1
        self.run = Run(self.runs, driver, status["task"], now, self.limits)
        print(f"[{WHO}] run {self.runs}: Dream-1 drives on {self.run.task!r}")
        self.report(now)

    async def watch(self, run: Run, now: float) -> None:
        frame = self.obs.get_video_frame(self.camera)
        if frame is not None and run.clip.wants(now):
            run.clip.add(now, shrink(frame))
        state = self.obs.get_state()
        verdict = None if state is None else run.add_pose(now, pose(state.values))

        if self.scoring is not None and self.scoring[2].done():
            scored, asked_at, future = self.scoring
            self.scoring, self.scored_at = None, asked_at
            try:
                p, log_p = future.result()
            except Exception as exc:
                print(f"[{WHO}] scoring failed: {exc!r}")
            else:
                if scored is run:
                    verdict = run.add_score(asked_at, p) or verdict
                    held = run.done_for(asked_at)
                    print(f"[{WHO}] run {run.number} at {asked_at - run.started:.0f}s: {p:.0%} done "
                          f"(log P(True) {log_p:.2f}), {run.trend() or 'first score'}"
                          + ("" if held is None or run.verdict else
                             f"; above {self.limits.success:.0%} for {held:.0f}s of {self.limits.hold_s:.0f}s"))
                    self.report(now)
        if self.scoring is None and len(run.clip) >= MIN_FRAMES and now - self.scored_at >= self.limits.every_s:
            frames, fps = run.clip.sample(VLM_FRAMES)
            self.scoring = (run, now, asyncio.ensure_future(self.worker.run(self.scorer.score, frames, fps, run.task)))

        if verdict:
            print(f"[{WHO}] run {run.number}: {verdict}: {run.reason}"
                  + ("; dry run, so Dream-1 keeps control" if self.dry_run else ""))
            if self.dry_run:
                self.report(now)
        if run.verdict and not self.dry_run:
            await self.release(run, now)  # a dry run keeps watching, so the log shows the scores that follow

    async def release(self, run: Run, now: float) -> None:
        """After a verdict, take control from Dream-1 so the arms hold, and tell the voice agent. Retried every tick
        until it works."""
        if self.obs.active_operator() == run.operator:
            try:
                await self.obs.set_active_operator(None)
            except Exception as exc:
                print(f"[{WHO}] could not take control from {run.operator}: {exc}; trying again")
                return
            run.released = True
            print(f"[{WHO}] took control from {run.operator}: the arms hold; 'continue' or c resumes")
        run.ended = now
        self.report(now)

    def report(self, now: float) -> None:
        """Send the voice agent the current report, without waiting; only the newest unsent one is kept."""
        self.reported_at = now
        run = self.run
        report = dict(run.report(now) if run else {"state": "idle"}, dry_run=self.dry_run)
        if run is not None and run.verdict and run.ended is None and not self.dry_run:
            report["state"] = "watching"  # a verdict is news once control is released, or someone else took it
        self.pending = report
        if self.sending is None or self.sending.done():
            self.sending = asyncio.ensure_future(self._send())

    async def _send(self) -> None:
        while (report := self.pending) is not None:
            self.pending = None
            try:
                await self.obs.perform_rpc(MONITOR_REPORT_RPC, json.dumps(report), destination=VOICE_AGENT_IDENTITY,
                                           response_timeout_ms=3000)
                if not self.heard:
                    print(f"[{WHO}] the voice agent takes the reports")
                self.heard = True
            except Exception as exc:
                if self.heard is not False:
                    print(f"[{WHO}] no voice agent took the report ({exc}); still trying")
                self.heard = False

    async def close(self) -> None:
        if self.sending is not None:
            self.sending.cancel()
            await asyncio.gather(self.sending, return_exceptions=True)
        if self.scoring is not None:
            await asyncio.gather(self.scoring[2], return_exceptions=True)


async def main(dry_run: bool) -> None:
    load_env()
    url, room = required_env("LIVEKIT_URL"), required_env("LIVEKIT_ROOM")
    limits = Limits.from_env()
    vlm = env_str("MONITOR_VLM", "Qwen/Qwen3-VL-8B-Instruct")
    camera = env_str("MONITOR_CAMERA", "overhead_camera")
    cfg = portal_config(ObserverConfig, room)
    if camera not in cfg.video_tracks:
        raise RuntimeError(f"MONITOR_CAMERA={camera!r} is not one of portal.yaml's videos: {cfg.video_tracks}")

    worker = HardwareThread(WHO)  # one score at a time, off the event loop
    print(f"[{WHO}] loading {vlm} (the first run downloads it) ...")
    scorer = await worker.run(Scorer, vlm, os.environ.get("MONITOR_DEVICE"))
    print(f"[{WHO}] {vlm} on {scorer.device}")

    obs = Observer(cfg)
    monitor = Monitor(obs, scorer, worker, limits, camera, dry_run)
    try:
        # Returns after the first sync with the robot's clock: start the robot first.
        await obs.connect(url, mint_token(MONITOR_IDENTITY, room))
        print(f"[{WHO}] watching {camera} for Dream-1{' (dry run: never takes control)' if dry_run else ''}; "
              f"done after {limits.hold_s:.0f}s at {limits.success:.0%} or more, stuck after {limits.stuck_s:.0f}s "
              f"without progress or {limits.still_s:.0f}s still")
        async for _ in pace(TICK_HZ):
            await monitor.tick(time.monotonic())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n[{WHO}] stopping")
    finally:
        await monitor.close()
        worker.close()
        try:
            await obs.disconnect()
        finally:
            obs.close()


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report verdicts, but never take control from Dream-1")
    asyncio.run(main(parser.parse_args().dry_run))


if __name__ == "__main__":
    run()
