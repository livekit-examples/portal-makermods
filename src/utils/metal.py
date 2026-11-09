"""Two Metal followers with three cameras, and two Metal leaders, over Dream Labs' `robots` driver.

Each arm reads and takes seven values: joint1..joint6 in radians, then the gripper in mm (0 closed).
Followers are energized; leaders open read-only and are moved by hand. The arms have no brakes, so
the only switch-off is `Followers.park()`, which first walks them back to where they were opened.
See tutorial/05-robot-loop.md.
"""
from __future__ import annotations

import logging
import math
import platform
import time

import numpy as np

from utils.common import env_camera_id, env_float, env_int, env_str, required_env

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
GRIPPER_MAX_MM = 100.0  # the driver's gripper runs 0..1; carried as mm
LIMITS_DEG = ((-160, 160), (-180, 0), (0, 180), (-123, 81), (-85, 85), (-145, 145))  # joint1..joint6
LIMITS_RAD = [(math.radians(lo), math.radians(hi)) for lo, hi in LIMITS_DEG]
PARK_LAG_RAD = math.radians(10)  # a parking joint further behind than this is blocked
PARK_TOL_RAD = math.radians(3)  # this close to rest counts as arrived


def arm_fields(side: str) -> list[str]:
    """One arm's seven `portal.yaml` fields."""
    return [f"{side}_joint{i}.pos" for i in range(1, 7)] + [f"{side}_gripper.pos"]


def clamp_to_limits(action: dict[str, float]) -> dict[str, float]:
    """Every joint inside its soft limit, every gripper inside its stroke."""
    out = dict(action)
    for side in SIDES:
        for name, (lo, hi) in zip(arm_fields(side), LIMITS_RAD + [(0.0, GRIPPER_MAX_MM)]):
            if name in out:
                out[name] = min(max(out[name], lo), hi)
    return out


def _gripper_deg(side: str, which: str) -> float:
    """The gripper motor's angle with the jaws fully OPEN or CLOSED, per arm or shared."""
    raw = env_str(f"METAL_{side.upper()}_GRIPPER_{which}_DEG", env_str(f"METAL_GRIPPER_{which}_DEG", ""))
    if not raw:
        raise RuntimeError(f"{side} arm: set METAL_GRIPPER_{which}_DEG (tutorial/01-first-recording.md)")
    return float(raw)


class Arm:
    """One arm. `read()` returns None on a hiccup instead of raising."""

    def __init__(self, side: str, can_channel: str, leader: bool) -> None:
        self.side, self.can_channel, self.leader = side, can_channel, leader
        self.errors = 0
        self.energized = False
        self.rest: list[float] | None = None  # the joints when opened: where park() returns
        self._arm = None  # robots.followers.FollowerArm

    def connect(self, energize: bool | None = None) -> None:
        """Open the arm: energized for a follower, read-only for a leader, unless `energize` says otherwise."""
        try:
            from robots import config, metal, motor_arm, robstride
        except ImportError as exc:
            raise RuntimeError("dreamlabs is not installed: run `uv sync`") from exc
        energize = not self.leader if energize is None else energize
        cfg = config.ArmConfig(
            can_channel=self.can_channel,
            home=(0.0,) * 6,  # never homed: the loop starts from the measured pose
            gripper_open_deg=_gripper_deg(self.side, "OPEN"),
            gripper_closed_deg=_gripper_deg(self.side, "CLOSED"),
            gripper_hold_torque_nm=env_float("METAL_GRIPPER_HOLD_NM", 0.5),
        )
        self._arm = motor_arm.open_arm(metal.ARM, cfg, self.side, opener=robstride.open_can_bus, energize=energize)
        self.energized = energize
        if (reading := self.read()) is None:
            raise RuntimeError(f"{self.side} arm: no pose after opening {self.can_channel}")
        self.rest = reading[:6]

    def read(self) -> list[float] | None:
        try:
            state = self._arm.read()
            values = [float(x) for x in state.positions_rad] + [float(state.gripper) * GRIPPER_MAX_MM]
            if len(values) != 7 or not all(map(math.isfinite, values)):
                raise RuntimeError(f"bad reading {values}")
            return values
        except Exception as exc:
            self.errors += 1
            if self.errors == 1 or self.errors % 30 == 0:  # first failure, then about once a second
                logger.warning("%s arm: read failed (%s); %d so far", self.side, exc, self.errors)
            return None

    def command(self, values: list[float]) -> None:
        from robots import types

        gripper = min(max(values[6] / GRIPPER_MAX_MM, 0.0), 1.0)
        self._arm.write(types.ArmCommand(positions_rad=np.asarray(values[:6], dtype=np.float64), gripper=gripper))

    def switch_off(self) -> None:
        """The driver's torque off, which parks where the arm was opened first. Call it once the arm is there."""
        self._arm.torque_off()
        self.energized = False

    def close(self) -> None:
        """Release the driver; an energized arm keeps holding its last target."""
        if self._arm is not None:
            self._arm.close()
            self._arm = None


class Camera:
    """A USB camera that serves its last good frame through a hiccup."""

    def __init__(self, name: str, cam) -> None:
        self.name, self._cam, self._last, self.errors = name, cam, None, 0

    def read(self):
        try:
            self._last = self._cam.read_latest(max_age_ms=500)
        except Exception as exc:
            self.errors += 1
            if self.errors == 1 or self.errors % 30 == 0:
                logger.warning("%s: %s; serving the last frame (%d so far)", self.name, exc, self.errors)
        return self._last


class StepLimiter:
    """Moves each joint target at most `max_step` rad per command, so an arm never jumps.

    It ramps the TARGET and never clamps the tracking error: under impedance control the error is
    the torque, and capping it can leave a joint stuck. The ramp restarts from the measured pose
    after a pause in commands. Grippers are not ramped.
    """

    MAX_LAG = 0.5  # rad the target may lead a joint that cannot follow
    PAUSE_S = 0.5

    def __init__(self, max_step: float) -> None:
        self.max_step = max_step
        self.present: dict[str, float] = {}
        self._target: dict[str, float] = {}
        self._last = 0.0
        self.clamped = 0

    def step(self, action: dict[str, float]) -> dict[str, float]:
        now = time.monotonic()
        if now - self._last > self.PAUSE_S:
            self._target = {}
        self._last = now
        if self.max_step <= 0:
            return dict(action)
        out = {}
        for name, goal in action.items():
            present = self.present.get(name)
            if present is None or name.endswith("_gripper.pos"):
                out[name] = goal
                continue
            base = self._target.get(name, present)
            target = base + min(max(goal - base, -self.max_step), self.max_step)
            target = min(max(target, present - self.MAX_LAG), present + self.MAX_LAG)
            self.clamped += target != goal
            self._target[name] = out[name] = target
        return out


def _read_pair(arms: dict[str, Arm]) -> dict[str, float]:
    """Both arms as one dict, or {} if either is unreadable: never half a row."""
    values = {}
    for side, arm in arms.items():
        if (reading := arm.read()) is None:
            return {}
        values.update(zip(arm_fields(side), reading))
    return values


class Followers:
    """Both followers and the cameras, behind `get_observation()` and `send_action()`.

    Parked, the arms are switched off at rest and open read-only: state keeps flowing, actions are ignored.
    """

    park_speed_env = "METAL_PARK_DEG_S"

    def __init__(self, arms: dict[str, Arm], cameras: dict[str, Camera], limiter: StepLimiter) -> None:
        self.arms, self.cameras, self.limiter = arms, cameras, limiter
        self.parked = False

    def connect(self) -> None:
        for camera in self.cameras.values():
            camera._cam.connect()
        self._open_arms(energize=True)

    def _open_arms(self, energize: bool) -> None:
        """Open every arm, or none: an arm already energized when another fails is switched off where it
        was opened, so the next start does not find it powered."""
        try:
            for arm in self.arms.values():
                arm.connect(energize)
        except Exception:
            for arm in self.arms.values():
                try:
                    if arm.energized:
                        arm.switch_off()
                    arm.close()
                except Exception as exc:
                    logger.warning("%s arm: could not switch off after a failed open (%s); support it", arm.side, exc)
            raise

    def get_observation(self) -> tuple[dict[str, float], dict]:
        """(state, frames). State is {} while an arm is unreadable; frames keep flowing."""
        state = _read_pair(self.arms)
        self.limiter.present.update(state)
        frames = {name: frame for name, cam in self.cameras.items() if (frame := cam.read()) is not None}
        return state, frames

    def send_action(self, action: dict[str, float]) -> None:
        if self.parked:
            return
        target = clamp_to_limits(self.limiter.step(action))
        for side, arm in self.arms.items():
            fields = arm_fields(side)
            if all(name in target and name in self.limiter.present for name in fields):  # never before a read
                arm.command([target[name] for name in fields])

    def go_home(self, fps: int) -> None:
        """Walk both arms back to where they were opened and hold them there, powered. Grippers stay as they are.

        Blocking: run it on the hardware thread. Raises, leaving the arms holding, if they are parked,
        an arm is unreadable, blocked (more than PARK_LAG_RAD behind), or short of its rest pose.
        """
        if self.parked:
            raise RuntimeError("parked: the arms are off at rest")
        try:
            self._walk_to_rest(fps)
        finally:
            self.limiter.present.clear()  # the arms moved under the limiter: never command before a fresh read

    def _walk_to_rest(self, fps: int) -> None:
        speed = env_float(self.park_speed_env, 10.0)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError(f"{self.park_speed_env} must be positive and finite")
        step = math.radians(speed) / fps
        present = {side: arm.read() for side, arm in self.arms.items()}
        if any(value is None for value in present.values()):
            raise RuntimeError("an arm is unreadable; not moving")
        target = {side: value[:6] for side, value in present.items()}
        settled = 0
        while settled < fps:  # one second at rest after the targets arrive
            started = time.monotonic()
            arrived = True
            for side, arm in self.arms.items():
                target[side] = [t + min(max(r - t, -step), step) for t, r in zip(target[side], arm.rest)]
                arrived &= all(abs(r - t) < 1e-9 for t, r in zip(target[side], arm.rest))
                arm.command(target[side] + [present[side][6]])
                if (reading := arm.read()) is None:
                    raise RuntimeError(f"{side} arm became unreadable on the way to rest; holding under power")
                if (lag := max(abs(t - q) for t, q in zip(target[side], reading[:6]))) > PARK_LAG_RAD:
                    arm.command(reading)
                    raise RuntimeError(f"{side} arm is {math.degrees(lag):.0f} deg behind its target on the way "
                                       "to rest (blocked?); holding where it is")
            settled = settled + 1 if arrived else 0
            time.sleep(max(0.0, 1 / fps - (time.monotonic() - started)))
        for side, arm in self.arms.items():
            reading = arm.read()
            if reading is None or max(abs(r - q) for r, q in zip(arm.rest, reading[:6])) > PARK_TOL_RAD:
                raise RuntimeError(f"{side} arm did not reach its rest pose; holding where it is")

    def park(self, fps: int) -> None:
        """Walk both arms home, switch them off, and reopen them read-only.

        Blocking: run it on the hardware thread. Raises, leaving the arms holding, wherever `go_home()` does.
        """
        if self.parked:
            return
        self.go_home(fps)
        for arm in self.arms.values():
            arm.switch_off()
        self.parked = True
        for arm in self.arms.values():
            arm.close()
        self._open_arms(energize=False)

    def unpark(self) -> None:
        """Power the parked arms up, holding where they rest (the new park pose)."""
        for arm in self.arms.values():
            arm.close()
        try:
            self._open_arms(energize=True)
        except Exception:
            self._open_arms(energize=False)  # stay parked and readable
            raise
        self.limiter.present.clear()  # never command before a fresh read
        self.parked = False

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            camera._cam.disconnect()
        for arm in self.arms.values():
            arm.close()


class Leaders:
    """Both leaders read-only, moved by hand, read as one Portal action."""

    def __init__(self, arms: dict[str, Arm]) -> None:
        self.arms = arms

    def connect(self) -> None:
        for arm in self.arms.values():
            arm.connect()

    def get_action(self) -> dict[str, float]:
        return _read_pair(self.arms)

    def disconnect(self) -> None:
        for arm in self.arms.values():
            arm.close()


def _arms(role: str) -> dict[str, Arm]:
    """One arm per side, on the link `METAL_<ROLE>_<SIDE>_CAN` names; required, so arms never swap silently."""
    return {side: Arm(side, required_env(f"METAL_{role}_{side.upper()}_CAN"), leader=role == "LEADER")
            for side in SIDES}


def build_followers(tracks: list[str], fps: int) -> Followers:
    """The followers, and one camera per video track: `<name>_camera` opens `METAL_CAM_<NAME>`."""
    from lerobot.cameras import Cv2Backends
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cameras = {}
    for i, name in enumerate(tracks):
        config = OpenCVCameraConfig(
            index_or_path=env_camera_id(f"METAL_CAM_{name.removesuffix('_camera').upper()}", f"/dev/video{2 * i}"),
            fps=fps, width=env_int("METAL_CAM_WIDTH", 640), height=env_int("METAL_CAM_HEIGHT", 480), fourcc="MJPG",
            # Other OpenCV backends can ignore the requested format and size on Linux.
            backend=Cv2Backends.V4L2 if platform.system() == "Linux" else Cv2Backends.ANY,
        )
        cameras[name] = Camera(name, OpenCVCamera(config))
    return Followers(_arms("FOLLOWER"), cameras, StepLimiter(env_float("METAL_MAX_STEP_RAD", 0.087)))


def build_leaders() -> Leaders:
    return Leaders(_arms("LEADER"))


def check_arms() -> None:
    """Print the followers' joints twice a second, motors off, until ctrl-c: move an arm to see its link."""
    arms = _arms("FOLLOWER")
    for arm in arms.values():
        arm.connect(energize=False)
    try:
        while True:
            for side, arm in arms.items():
                if (values := arm.read()) is not None:
                    joints = " ".join(f"{math.degrees(v):+5.0f}" for v in values[:6])
                    print(f"{side:5s} on {arm.can_channel}: joints {joints} deg, gripper {values[6]:3.0f} mm")
            print()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for arm in arms.values():
            arm.close()
