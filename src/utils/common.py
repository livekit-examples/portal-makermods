"""Shared plumbing: env loading, LiveKit tokens, and the tick pacer.

`portal.yaml` and the `.env` files live two directories above this file, at the repo root.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Optional

os.environ.setdefault("RUST_LOG", "warn,livekit_portal::sync_buffer=error,libwebrtc=error")

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.room import RoomConfiguration

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "portal.yaml"


def portal_config(config_type, room: str, profile: str | None = None):
    """Read one rig's wire contract from portal.yaml; observers use PORTAL_PROFILE."""
    import yaml

    document = yaml.safe_load(CONFIG_PATH.read_text())
    profiles = document.pop("profiles", {})
    profile = profile or env_str("PORTAL_PROFILE", "metal")
    if profile != "metal":
        if profile not in profiles:
            raise ValueError(f"unknown Portal profile {profile!r}")
        document.update(profiles[profile])
    return config_type.from_yaml_str(yaml.safe_dump(document), room)


ROBOT_IDENTITY = "robot"
TELEOPERATOR_IDENTITY = "teleoperator"
HAND_TELEOP_IDENTITY = "hand-teleop"  # the webcam hand teleop's default; see hand_teleop_identity()
RECORDER_IDENTITY = "recorder"
PARK_RPC = "park"  # `p` in any operator: the robot parks the arms, or powers parked arms up
PARK_TOGGLE = ""  # PARK_RPC payload for `p`: park, or power up parked arms
PARK_ONLY = "park"  # PARK_RPC payload that only parks: never powers up, and parked arms stay as they are
HOME_RPC = "home"  # the robot walks the powered arms back to rest and holds them there: never parks or powers up

# The task-switching Dream-1 (`uv run dream-1-tasks`) answers these; the voice agent calls them.
# Each takes and returns JSON; every reply is the status (see src/policies/dream_1_tasks/run.py).
DREAM_TASKS_IDENTITY = "policy-dream-1-tasks"
DREAM_STATUS_RPC = "dream.status"
DREAM_SET_TASK_RPC = "dream.set_task"  # {"task": "put the red cube in the box"}; pauses first if Dream-1 drives
DREAM_PAUSE_RPC = "dream.pause"  # Dream-1 releases control; the arms hold
DREAM_RESUME_RPC = "dream.resume"  # Dream-1 takes control, starting a new episode on the current task
DREAM_TO_TELEOP_RPC = "dream.to_teleop"  # control passes to the teleoperator's leader arms
DREAM_TO_HAND_TELEOP_RPC = "dream.to_hand_teleop"  # control passes to the webcam hand teleop
DREAM_STOP_HAND_TELEOP_RPC = "dream.stop_hand_teleop"  # the hand teleop loses control: no operator; the arms hold

# The task monitor (`uv run task-monitor`) watches dream-1-tasks drive and calls this on the voice agent with its
# report as JSON, every few seconds and at once on a verdict (see src/task_monitor.py).
VOICE_AGENT_IDENTITY = "voice-agent"
MONITOR_IDENTITY = "task-monitor"
MONITOR_REPORT_RPC = "monitor.report"


async def request_park(op, who: str) -> None:
    """Ask the robot to park or power up; it answers at once and acts on its own loop."""
    try:
        print(f"[{who}] robot: {await op.perform_rpc(PARK_RPC, destination=ROBOT_IDENTITY)}")
    except Exception as exc:
        print(f"[{who}] park request failed: {exc}; press p to retry")


async def claim_or_release(op, who: str) -> None:
    """A policy's own `c`: take control, or give it up (the robot then holds)."""
    me = op.local_identity()
    try:
        await op.set_active_operator(None if op.active_operator() == me else me)
    except Exception as exc:
        print(f"[{who}] could not change control: {exc}; press c to retry")


def load_env() -> None:
    """Load `.env` then `.env.local` (which overrides) from the repo root and cwd, then set up logging."""
    for directory in (ROOT, pathlib.Path.cwd()):
        for name, override in ((".env", False), (".env.local", True)):
            if (path := directory / name).exists():
                load_dotenv(path, override=override)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())


def required_env(name: str) -> str:
    if not (value := os.environ.get(name)):
        raise RuntimeError(f"{name} must be set (see .env.example)")
    return value


def env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name) or default)


def hand_teleop_identity() -> str:
    """The webcam hand teleop's identity: `HAND_TELEOP_IDENTITY`, else "hand-teleop". Read after `load_env()`."""
    return env_str("HAND_TELEOP_IDENTITY", HAND_TELEOP_IDENTITY)


def env_camera_id(name: str, default: str) -> int | str:
    """Camera identifier: an int index on macOS, a /dev/videoN string on Linux."""
    raw = env_str(name, default)
    return int(raw) if raw.isdigit() else raw


def mint_token(identity: str, room: str, ttl_hours: int = 6) -> str:
    """Mint a LiveKit JWT scoped to `room` with low-latency playout bounds.

    `can_update_own_metadata` is required so Portal can publish its
    `lk.portal.role` attribute on connect, which peer discovery reads.
    """
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_update_own_metadata=True,
    )
    return (
        api.AccessToken(required_env("LIVEKIT_API_KEY"), required_env("LIVEKIT_API_SECRET"))
        .with_identity(identity)
        .with_grants(grants)
        .with_room_config(RoomConfiguration(name=room, min_playout_delay=0, max_playout_delay=1))
        .with_ttl(datetime.timedelta(hours=ttl_hours))
        .to_jwt()
    )


async def pace(fps: int) -> AsyncIterator[int]:
    """Yield a tick index at `fps`, as `async for tick in pace(30)`.

    A late tick moves the schedule forward instead of piling up drift. Even
    then it yields with `sleep(0)`: without that, a body that always overruns
    would starve the Portal callbacks it depends on.
    """
    interval = 1.0 / fps
    next_tick = time.perf_counter()
    tick = 0
    while True:
        yield tick
        tick += 1
        next_tick += interval
        now = time.perf_counter()
        if next_tick > now:
            await asyncio.sleep(next_tick - now)
        else:
            next_tick = now
            await asyncio.sleep(0)


class HardwareThread:
    """Runs one device's blocking calls on one dedicated thread.

    The event loop stays free while the device works, and a single worker
    means two calls can never overlap on the device's serial bus.
    """

    def __init__(self, name: str) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def close(self) -> None:
        """Wait for the call in flight, so the device is never disconnected mid-transaction."""
        self._pool.shutdown(wait=True)


class TickBudget:
    """Tick timings behind the robot's 5s diagnostic line.

    `device_ms` is time inside the driver calls and `tick_ms` the whole body, so
    a slow CAN bus and slow publishing show up separately.
    """

    def __init__(self, fps: int) -> None:
        self.fps = fps
        self.reset()

    def reset(self) -> None:
        self.device_ms: list[float] = []
        self.tick_ms: list[float] = []
        self.started = time.perf_counter()

    def record(self, device_ms: float, tick_ms: float) -> None:
        self.device_ms.append(device_ms)
        self.tick_ms.append(tick_ms)

    def report(self) -> Optional[str]:
        ticks = len(self.tick_ms)
        if not ticks:
            return None
        budget_ms = 1000.0 / self.fps
        over = sum(ms > budget_ms for ms in self.tick_ms)
        line = (f"{ticks / (time.perf_counter() - self.started):.1f} fps of {self.fps} requested; "
                f"device {sum(self.device_ms) / ticks:.0f}ms (worst {max(self.device_ms):.0f}ms), "
                f"tick {sum(self.tick_ms) / ticks:.0f}ms (worst {max(self.tick_ms):.0f}ms) "
                f"of {budget_ms:.0f}ms budget; {over}/{ticks} over")
        self.reset()
        return line
