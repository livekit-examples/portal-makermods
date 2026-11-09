"""Dream Labs' Dream-1 as a Portal operator. See tutorial/02-dream-1.md.

    uv run dream-1 --task "fold the towel" --dry-run

Dream-1 runs on Dream Labs' servers: it reads three views and both arms and answers chunks of
joint rows, through the `dream` client's model of the Metal arm. Keys in this terminal: `c`
takes or releases control (as `c` in teleop does), `p` parks the arms or powers them up.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time
from pathlib import Path

import cv2
import dream
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.common import (portal_config, HardwareThread, claim_or_release, env_float, env_int, env_str, load_env,
                          mint_token, pace, request_park, required_env)
from livekit.portal import Operator, OperatorConfig, frame_bytes_to_numpy_rgb
from utils.hotkeys import Hotkeys
from utils.metal import GRIPPER_MAX_MM, SIDES, arm_fields, clamp_to_limits

IDENTITY = "policy-dream-1"
CAMERAS = {"overhead": "overhead_camera", "wrist_l": "left_wrist_camera", "wrist_r": "right_wrist_camera"}
ROW_HZ = 20  # chunk rows played per second
STALE_S = 0.2  # hold when the newest observation is older than this


def embodiment() -> dict:
    """Where each part sits in a state or action row; the client checks every request and reply against it."""
    spans, start = {}, 0
    for side in ("right", "left"):
        for part, width in (("arm", 6), ("grip", 1), ("eef_xyz", 3), ("eef_rot6", 6)):
            spans[f"{side}_{part}"] = [start, start + width]
            start += width
    return {"state": spans, "action": spans, "video": {role: f"observation.images.{role}" for role in CAMERAS}}


def encode_frame(rgb: np.ndarray) -> str:
    """Center-crop to 16:9, resize to 320x176, and encode as base64 JPEG."""
    h, w = rgb.shape[:2]
    if h * 16 > w * 9:
        top = (h - w * 9 // 16) // 2
        rgb = rgb[top:top + w * 9 // 16]
    small = cv2.resize(rgb, (320, 176), interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpeg.tobytes()).decode("ascii")


def to_model(state: dict) -> np.ndarray:
    """Portal state -> 14 model values: per arm (left, right) six joints in radians, then grip 0..1."""
    row = []
    for side in SIDES:
        *joints, gripper = arm_fields(side)
        row += [state[name] for name in joints] + [min(max(state[gripper] / GRIPPER_MAX_MM, 0.0), 1.0)]
    return np.array(row)


def to_robot(row: np.ndarray) -> dict[str, float]:
    """14 model values -> a Portal action, inside the soft limits."""
    action = dict(zip(arm_fields("left") + arm_fields("right"), map(float, row)))
    for side in SIDES:
        action[f"{side}_gripper.pos"] *= GRIPPER_MAX_MM
    return clamp_to_limits(action)


def chunk_rows(reply: dict) -> np.ndarray:
    """The reply's two chunks side by side, left then right: [rows, 14]."""
    left, right = (np.asarray(reply[key], dtype=np.float64) for key in ("q_l_chunk", "q_r_chunk"))
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 7 or not np.isfinite([left, right]).all():
        raise dream.DreamActionError(f"chunks of shape {left.shape} and {right.shape}; expected [rows, 7] each")
    return np.concatenate([left, right], axis=1)


def retry_delay(error: Exception) -> float:
    """Seconds to wait before asking again; raises when asking again cannot help (a bad key, a bad reply)."""
    if isinstance(error, dream.DreamSessionExpired):
        return 0.0  # the client already opened a new session
    if isinstance(error, dream.DreamError) and not (
            isinstance(error, (dream.DreamTimeout, dream.DreamProtocolError)) or getattr(error, "retryable", False)):
        raise error
    print(f"[dream-1] {getattr(error, 'reason', None) or error!r}; asking again")  # incl. network errors
    return getattr(error, "retry_after_s", None) or 1.0


async def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=env_str("PORTAL_DATA_TASK", "MakerMods Metal bimanual manipulation"))
    parser.add_argument("--dry-run", action="store_true", help="print the targets instead of sending them")
    args = parser.parse_args()

    url, room = required_env("LIVEKIT_URL"), required_env("LIVEKIT_ROOM")
    cfg = portal_config(OperatorConfig, room, "metal")
    fps = env_int("PORTAL_FPS", cfg.fps)
    # Per-tick cap on the commanded target: joints in rad, grips 5% of their range per row.
    max_step = np.tile([env_float("DREAM_MAX_JOINT_RAD_S", 0.5) / fps] * 6 + [0.05 * ROW_HZ / fps], 2)

    client = dream.DreamClient()  # reads DREAM_API_KEY from .env (or ~/.dreamlabs/credentials)
    worker = HardwareThread("dream-1")  # one request at a time, off the event loop
    await worker.run(client.policy, {
        "checkpoint": env_str("DREAM_CHECKPOINT", "dreamlabs/dream-1-beta"),
        "embodiment": embodiment(),
        "ik": {"arm": "metal", "head": "eef"},
        "action_hz": ROW_HZ,
    })

    def ask() -> tuple[float, np.ndarray]:
        asked_at = time.monotonic()
        return asked_at, chunk_rows(client.get_action())

    def send_observation(obs) -> None:
        frames = {}
        for role, track in CAMERAS.items():
            frame = obs.frames[track]
            frames[f"video.{role}"] = encode_frame(frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height))
        row = to_model(obs.state)
        client.set_observation(frames, {
            "prompt": args.task,
            "state.left_arm": row[:6].tolist(), "state.left_grip": row[6:7].tolist(),
            "state.right_arm": row[7:13].tolist(), "state.right_grip": row[13:].tolist(),
        })

    op = Operator(cfg)
    latest_obs, received_at = None, 0.0
    control_changed = True

    def on_observation(obs) -> None:
        nonlocal latest_obs, received_at
        latest_obs, received_at = obs, time.monotonic()

    def on_active_operator_changed(identity) -> None:
        nonlocal control_changed
        control_changed = True
        print(f"[dream-1] active operator: {identity}")

    op.on_observation(on_observation)
    op.on_active_operator_changed(on_active_operator_changed)

    request = None  # the get_action() in flight
    chunk = None  # (asked_at, rows, first row played); row k is due k / ROW_HZ s after asked_at
    target = None  # the last commanded row
    sent_ts = None  # the observation the client has
    retry_at = 0.0
    hotkeys = Hotkeys(global_tap=False)  # this terminal only; works over SSH
    try:
        await op.connect(url, mint_token(IDENTITY, room))
        hotkeys.start()
        print(f"[dream-1] ready{' (dry run)' if args.dry_run else ''}; c takes control, p parks")
        async for tick in pace(fps):
            keys = hotkeys.drain()
            if "p" in keys:
                await request_park(op, "dream-1")
            if "c" in keys:
                await claim_or_release(op, "dream-1")
            obs = latest_obs
            if op.active_operator() != op.local_identity():
                continue
            if obs is None or time.monotonic() - received_at > STALE_S or set(CAMERAS.values()) - set(obs.frames):
                continue  # no fresh view: send nothing, and the arms hold

            if control_changed:
                # Each handoff starts a new episode, ramping from where the arms are now.
                control_changed = False
                if request is not None:
                    await asyncio.gather(request, return_exceptions=True)
                    request = None
                await worker.run(client.reset_episode)
                chunk, sent_ts, retry_at = None, None, 0.0
                target = to_model(obs.state)

            if obs.timestamp_us != sent_ts:
                send_observation(obs)
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
            action = to_robot(target)
            if not args.dry_run:
                op.send_action(action, in_reply_to_ts_us=obs.timestamp_us)
            elif tick % fps == 0:
                print("[dream-1] " + ", ".join(f"{k}={v:.2f}" for k, v in action.items()))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[dream-1] stopping")
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
