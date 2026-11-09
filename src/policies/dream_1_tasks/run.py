"""Dream-1 with a task you can change while it runs. See tutorial/10-voice-tasks.md.

    uv run dream-1-tasks --task "put the blue cube in the box" --task "put the red cube in the box" --dry-run

This app drives one Seeed B601-DM with overhead and arm cameras (tutorial/13-rebot.md).
The scheduling follows `uv run dream-1`; RebotMapping supplies this arm's units and kinematics.
`TaskControl` holds the task: keys `1`-`9` pick a preset, and other peers, over RPC, change it,
pause and resume Dream-1, hand the arms to the teleoperator or the webcam hand teleop, or take them away from the
hand teleop. The voice agent (`uv run voice-agent`) is one.

A task change while Dream-1 drives releases control first, so the arms hold until someone resumes: a new task
never starts motion. Each resume starts a new Dream-1 episode on the current task. With no task given at start,
Dream-1 has none and does not move, even when handed control, until one is set. Keys in this terminal: `c` takes
or releases control, `p` parks the arms or powers them up, `1`-`9` switch task.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import dream
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from livekit.portal import Operator, OperatorConfig, RpcError, frame_bytes_to_numpy_rgb
from policies.dream_1.run import ROW_HZ, STALE_S, encode_frame, retry_delay
from policies.dream_1_tasks.rebot import CAMERAS, RebotMapping, chunk_rows
from utils.common import (portal_config, DREAM_PAUSE_RPC, DREAM_RESUME_RPC, DREAM_SET_TASK_RPC, DREAM_STATUS_RPC,
                          DREAM_STOP_HAND_TELEOP_RPC, DREAM_TASKS_IDENTITY, DREAM_TO_HAND_TELEOP_RPC,
                          DREAM_TO_TELEOP_RPC, TELEOPERATOR_IDENTITY, HardwareThread, claim_or_release, env_float,
                          env_int, env_str, hand_teleop_identity, load_env, mint_token, pace, request_park,
                          required_env)
from utils.hotkeys import Hotkeys
from utils.session import MAX_TASKS, preset_tasks

WHO = "dream-1-tasks"
MAX_TASK_CHARS = 200  # one instruction; anything longer is more likely a transcription accident


class TaskControl:
    """The task Dream-1 is prompted with, and who drives, for keys here and RPCs from other peers.

    `changed` tells the loop to start a new episode, so a chunk planned for the old task is never played.
    """

    def __init__(self, op: Operator, presets: list[str]) -> None:
        self.op = op
        self.presets = presets
        self.task: str | None = presets[0] if presets else None  # None: nothing to do until a task is set
        self.changed = False
        self.hand_teleop = hand_teleop_identity()

    def driving(self) -> bool:
        return self.op.active_operator() == self.op.local_identity()

    def status(self) -> dict:
        """What every RPC answers: the task, the presets, who drives, and whether the teleoperator or the hand
        teleop could."""
        operators = self.op.operators()
        return {"task": self.task, "presets": self.presets, "driving": self.driving(),
                "active_operator": self.op.active_operator(),
                "teleoperator_connected": TELEOPERATOR_IDENTITY in operators,
                "hand_teleop": self.hand_teleop, "hand_teleop_connected": self.hand_teleop in operators}

    async def set_task(self, task: str, who: str) -> dict:
        """Prompt Dream-1 with `task` from now on. Releases control first if Dream-1 drives."""
        task = " ".join(str(task).split())
        if not task or len(task) > MAX_TASK_CHARS:
            raise ValueError(f"a task is one instruction of 1-{MAX_TASK_CHARS} characters")
        await self.pause(who)  # raises if control cannot be released: then the task stays as it was
        if task != self.task:
            self.task, self.changed = task, True
            print(f"[{WHO}] {who}: task {task!r}; c (or 'continue') resumes")
        return self.status()

    async def pause(self, who: str) -> dict:
        """Release control if Dream-1 has it; the robot then holds the arms where they are."""
        if self.driving():
            print(f"[{WHO}] {who}: pausing")
            await self._hand_to(None)
        return self.status()

    async def resume(self, who: str) -> dict:
        """Take control (from whoever has it), starting a new episode on the current task."""
        if self.task is None:
            raise ValueError("Dream-1 has no task yet; give it one first")
        if not self.driving():
            print(f"[{WHO}] {who}: resuming {self.task!r}")
            await self._hand_to(self.op.local_identity())
        return self.status()

    async def to_teleop(self, who: str) -> dict:
        """Hand control to the teleoperator, the person on the leader arms."""
        return await self._to_human(TELEOPERATOR_IDENTITY, "the teleoperator", "uv run teleoperator", who)

    async def to_hand_teleop(self, who: str) -> dict:
        """Hand control to the webcam hand teleop. The arms hold until a hand grips one."""
        return await self._to_human(self.hand_teleop, "the hand teleop", "uv run --group hands hand-teleop", who)

    async def stop_hand_teleop(self, who: str) -> dict:
        """Take control from the hand teleop, if it has it: then no operator drives and the robot holds the arms.
        Dream-1 or the leader arms driving are left alone."""
        if self.op.active_operator() == self.hand_teleop:
            print(f"[{WHO}] {who}: taking control from the hand teleop ({self.hand_teleop}); nobody drives")
            await self._hand_to(None)
        return self.status()

    async def _to_human(self, identity: str, name: str, start: str, who: str) -> dict:
        """Hand control to a human's operator, only one that is connected: else Dream-1 keeps what it has."""
        if identity not in self.op.operators():
            raise ValueError(f"{name} is not connected; start `{start}`")
        if self.op.active_operator() != identity:
            print(f"[{WHO}] {who}: handing control to {name} ({identity})")
            await self._hand_to(identity)
        return self.status()

    async def _hand_to(self, identity: str | None) -> None:
        await self.op.set_active_operator(identity)
        for _ in range(40):  # this side mirrors the robot's attribute, a moment after the robot answers
            if self.op.active_operator() == identity:
                return
            await asyncio.sleep(0.025)

    def register_rpcs(self) -> None:
        """Answer the `dream.*` methods in src/utils/common.py, each with the status as JSON."""
        def handler(change):
            async def handle(data) -> str:
                try:
                    return json.dumps(await change(json.loads(data.payload or "{}"), data.caller_identity))
                except Exception as exc:  # the caller reads the message, e.g. the voice agent's LLM
                    raise RpcError.Error(code=1500, message=str(exc), data=None) from exc
            return handle

        async def status(_payload, _who) -> dict:
            return self.status()

        async def set_task(payload, who) -> dict:
            return await self.set_task(payload.get("task", ""), who)

        self.op.register_rpc_method(DREAM_STATUS_RPC, handler(status))
        self.op.register_rpc_method(DREAM_SET_TASK_RPC, handler(set_task))
        self.op.register_rpc_method(DREAM_PAUSE_RPC, handler(lambda _payload, who: self.pause(who)))
        self.op.register_rpc_method(DREAM_RESUME_RPC, handler(lambda _payload, who: self.resume(who)))
        self.op.register_rpc_method(DREAM_TO_TELEOP_RPC, handler(lambda _payload, who: self.to_teleop(who)))
        self.op.register_rpc_method(DREAM_TO_HAND_TELEOP_RPC,
                                    handler(lambda _payload, who: self.to_hand_teleop(who)))
        self.op.register_rpc_method(DREAM_STOP_HAND_TELEOP_RPC,
                                    handler(lambda _payload, who: self.stop_hand_teleop(who)))


def session_tasks(given: list[str] | None) -> list[str]:
    """`--task` values, else `PORTAL_DATA_TASKS` (as in teleop), else `PORTAL_DATA_TASK`. Empty when none is set:
    then Dream-1 starts with no task."""
    single = env_str("PORTAL_DATA_TASK", "").strip()
    return (given or preset_tasks() or ([single] if single else []))[:MAX_TASKS]


async def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="a preset task; repeat for keys 1-9 (the first starts)")
    parser.add_argument("--dry-run", action="store_true", help="print the targets instead of sending them")
    args = parser.parse_args()

    url, room = required_env("LIVEKIT_URL"), required_env("LIVEKIT_ROOM")
    cfg = portal_config(OperatorConfig, room, "rebot")
    mapping = RebotMapping(cfg, required_env("REBOT_ARM_FILE"))
    fps = env_int("PORTAL_FPS", cfg.fps)
    # Per-tick cap on the commanded target: joints in rad, grips 5% of their range per row.
    max_step = np.array([env_float("DREAM_MAX_JOINT_RAD_S", 0.5) / fps] * 6 + [0.05 * ROW_HZ / fps])

    client = dream.DreamClient()  # reads DREAM_API_KEY from .env (or ~/.dreamlabs/credentials)
    worker = HardwareThread("dream-1")  # one request at a time, off the event loop
    policy = mapping.policy_config(env_str("DREAM_CHECKPOINT", "dreamlabs/dream-1-beta"), ROW_HZ)

    def ask() -> tuple[float, np.ndarray]:
        asked_at = time.monotonic()
        return asked_at, chunk_rows(client.get_action())

    def send_observation(obs, task: str) -> None:
        frames = {"video.wrist_l": mapping.blank_wrist}
        for role, track in CAMERAS.items():
            frame = obs.frames[track]
            frames[f"video.{role}"] = encode_frame(frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height))
        client.set_observation(frames, mapping.observation(obs.state, task))

    op = Operator(cfg)
    tasks = TaskControl(op, session_tasks(args.task))
    tasks.register_rpcs()
    latest_obs, received_at = None, 0.0
    control_changed = True

    def on_observation(obs) -> None:
        nonlocal latest_obs, received_at
        latest_obs, received_at = obs, time.monotonic()

    def on_active_operator_changed(identity) -> None:
        nonlocal control_changed
        control_changed = True
        print(f"[{WHO}] active operator: {identity}")

    op.on_observation(on_observation)
    op.on_active_operator_changed(on_active_operator_changed)

    request = None  # the get_action() in flight
    chunk = None  # (asked_at, rows, first row played); row k is due k / ROW_HZ s after asked_at
    target = None  # the last commanded row
    sent_ts = None  # the observation the client has
    retry_at = 0.0
    hotkeys = Hotkeys(global_tap=False)  # this terminal only; works over SSH
    try:
        await worker.run(client.policy, policy)
        await op.connect(url, mint_token(DREAM_TASKS_IDENTITY, room))
        hotkeys.start()
        for number, task in enumerate(tasks.presets, start=1):
            print(f"[{WHO}]   {number}  {task}")
        on = f"on {tasks.task!r}" if tasks.task else "with no task (the voice agent sets one)"
        switch = f", 1-{len(tasks.presets)} switch task" if tasks.presets else ""
        print(f"[{WHO}] ready{' (dry run)' if args.dry_run else ''} {on}; c takes control, p parks{switch}")
        async for tick in pace(fps):
            for key in hotkeys.drain():
                if key == "p":
                    await request_park(op, WHO)
                elif key == "c" and tasks.task is None and not tasks.driving():
                    print(f"[{WHO}] no task yet: set one by voice, or restart with --task")
                elif key == "c":
                    await claim_or_release(op, WHO)
                elif key.isdigit() and int(key) <= len(tasks.presets):
                    try:
                        await tasks.set_task(tasks.presets[int(key) - 1], "this terminal")
                    except Exception as exc:
                        print(f"[{WHO}] could not switch task: {exc}; press {key} to retry")
            obs = latest_obs
            if not tasks.driving() or tasks.task is None:
                continue  # with no task, even when handed control, send nothing: the arms hold
            if (obs is None or time.monotonic() - received_at > STALE_S
                    or set(CAMERAS.values()) - set(obs.frames) or set(mapping.fields) - set(obs.state)):
                continue  # no fresh view: send nothing, and the arms hold

            if control_changed or tasks.changed:
                # Each handoff or task starts a new episode, ramping from where the arms are now.
                control_changed = tasks.changed = False
                if request is not None:
                    await asyncio.gather(request, return_exceptions=True)
                    request = None
                await worker.run(client.reset_episode)
                chunk, sent_ts, retry_at = None, None, 0.0
                target = mapping.to_model(obs.state)

            if obs.timestamp_us != sent_ts:
                send_observation(obs, tasks.task)
                sent_ts = obs.timestamp_us

            now = time.monotonic()
            if request is not None and request.done():
                try:
                    asked_at, rows = request.result()
                    chunk = (asked_at, rows, int((now - asked_at) * ROW_HZ))  # skip rows the round trip used up
                except Exception as error:
                    retry_at = now + retry_delay(error)
                request = None

            row, played, remaining = None, 0, 0
            if chunk is not None:
                asked_at, rows, first = chunk
                due = int((now - asked_at) * ROW_HZ)
                if due < len(rows):
                    row, played, remaining = rows[due], due - first, len(rows) - due
            # Ask for the next chunk early, so it lands before this one runs out.
            if request is None and now >= retry_at and client.refill_due(played, remaining):
                request = asyncio.ensure_future(worker.run(ask))
            if row is None:
                continue

            target = target + np.clip(row - target, -max_step, max_step)
            action = mapping.to_robot(target)
            if not args.dry_run:
                op.send_action(action, in_reply_to_ts_us=obs.timestamp_us)
            elif tick % fps == 0:
                print(f"[{WHO}] {tasks.task!r}: " + ", ".join(f"{k}={v:.2f}" for k, v in action.items()))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n[{WHO}] stopping")
    finally:
        hotkeys.stop()
        worker.close()  # waits for a request in flight
        client.close()
        try:
            await op.disconnect()
        finally:
            op.close()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
