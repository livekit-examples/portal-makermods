"""Seeed B601-DM: six Damiao joints and a gripper, via MotorBridge.

Motor IDs/models and POS_VEL gains follow Seeed's reBotArm_control_py DM
configuration. Reads and writes use motor/URDF radians; Portal carries grip
as 0..1. All methods run on the robot's one hardware thread.
"""
from __future__ import annotations

import logging
import math
import platform

from utils.common import env_camera_id, env_float, env_int, env_str, required_env
from utils.metal import Camera, Followers, StepLimiter, arm_fields

logger = logging.getLogger(__name__)

# Seeed's rebotDM.yaml soft limits, not Metal's limits (notably joint3).
LIMITS = ((-2.6, 2.6), (-3.8, 0.0), (-3.8, 0.0), (-1.57, 1.57), (-1.57, 1.57), (-2.6, 2.6), (0.0, 1.0))
# A disabled arm can settle slightly past a boundary after parking at zero.
# Only startup reads get this margin; home and commanded targets stay inside LIMITS.
STARTUP_TOL_RAD = math.radians(0.5)


def clamp(values):
    if len(values) != 7 or not all(math.isfinite(v) for v in values):
        raise ValueError("ReBot needs seven finite values: six radians and grip 0..1")
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(values, LIMITS)]


def fields_from_config(cfg) -> list[str]:
    fields = [field.name for field in cfg.state_schema]
    if fields != arm_fields("right") or fields != [field.name for field in cfg.action_schema]:
        raise ValueError("ReBot needs matching state/action schemas in right joint1..6, gripper order")
    if set(cfg.video_tracks) != {"overhead_camera", "arm_camera"}:
        raise ValueError("ReBot needs overhead_camera and arm_camera")
    return fields


class Arm:
    """MotorBridge adapter with fresh bus round trips and no implicit switch-off."""

    side = "right"

    def __init__(self, channel: str, transport: str, closed: float, opened: float):
        if transport not in ("dm-serial", "socketcan"):
            raise ValueError("REBOT_TRANSPORT must be dm-serial or socketcan")
        if not all(math.isfinite(v) and -5.2 <= v <= 0 for v in (closed, opened)) or closed == opened:
            raise ValueError("ReBot gripper endpoints must differ and lie in [-5.2, 0] radians")
        self.channel, self.transport = channel, transport
        self.closed, self.opened = closed, opened
        self.errors, self.energized = 0, False
        self.rest = None
        self._gripper_rad = None  # raw feedback, before normalization hides out-of-range readings
        self.ctrl = None
        self.motors = []
        self.speed = env_float("REBOT_SPEED_RAD_S", 0.5)
        if not math.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("REBOT_SPEED_RAD_S must be positive and finite")

    def connect(self, energize=True):
        from motorbridge import Controller, Mode

        self.ctrl = (Controller.from_dm_serial(self.channel, 921600) if self.transport == "dm-serial"
                     else Controller(self.channel))
        try:
            self.motors = [self.ctrl.add_damiao_motor(i, 0x10 + i, "4340P" if i <= 3 else "4310")
                           for i in range(1, 8)]
            for i, motor in enumerate(self.motors, 1):
                if motor.get_register_u32(8, timeout_ms=100) != i:
                    raise RuntimeError(f"motor {i}: unexpected CAN ID")
            present = self.read()
            if present is None:
                raise RuntimeError("ReBot has no complete pose; check power, IDs, and calibration")
            outside = [f"joint{i}={value:+.6f} rad (limit [{lo:g}, {hi:g}])"
                       for i, (value, (lo, hi)) in enumerate(zip(present[:6], LIMITS[:6]), 1)
                       if not lo - STARTUP_TOL_RAD <= value <= hi + STARTUP_TOL_RAD]
            grip_lo, grip_hi = sorted((self.closed, self.opened))
            if not grip_lo - STARTUP_TOL_RAD <= self._gripper_rad <= grip_hi + STARTUP_TOL_RAD:
                outside.append(f"gripper={self._gripper_rad:+.6f} rad "
                               f"(calibrated range [{grip_lo:g}, {grip_hi:g}])")
            if outside:
                raise RuntimeError("ReBot startup pose is outside its soft limits plus 0.5 deg settling margin: "
                                   + "; ".join(outside) + "; check the resting pose and joint zero calibration")
            if any(motor.get_state().status_code != 0 for motor in self.motors):
                raise RuntimeError("ReBot is already powered; support it and cycle power before restarting")
            bounded = clamp(present)
            for i, (value, target) in enumerate(zip(present[:6], bounded[:6]), 1):
                if value != target:
                    logger.warning("ReBot joint%d at %+.6f rad is within the startup settling margin; "
                                   "home/hold target clamped to %+.6f rad", i, value, target)
            present = bounded
            self.rest = present[:6]
            if energize:
                for i, motor in enumerate(self.motors):
                    gains = (0.0125, 0.004, 150.0, 0.5) if i < 3 else (0.0008, 0.002, 50.0, 1.0)
                    for register, gain in zip((25, 26, 27, 28), gains):
                        motor.write_register_f32(register, gain)
                    motor.ensure_mode(Mode.POS_VEL, timeout_ms=1000)
                # Seed each position controller at the measured pose before enabling.
                self.command(present)
                self.energized = True  # includes a partially successful enable: never drop it on a fault
                self.ctrl.enable_all()
                self.command(present)
        except BaseException:
            self.close()  # no disable on an unknown/partially powered pose
            raise

    def read(self):
        try:
            values = []
            for i, motor in enumerate(self.motors, 1):
                motor.request_feedback()
                # get_state() alone is cached. Register 80 is the live motor
                # position (also used by MotorBridge to seed a mode switch).
                position = motor.get_register_f32(80, timeout_ms=100)
                state = motor.get_state()
                if state is None or state.status_code not in (0, 1) or (self.energized and state.status_code != 1):
                    raise RuntimeError(f"motor {i}: missing, disabled, or faulted feedback")
                values.append(float(position))
            if len(values) != 7 or not all(map(math.isfinite, values)):
                raise RuntimeError("incomplete or nonfinite joint feedback")
            self._gripper_rad = values[6]
            values[6] = min(max((values[6] - self.closed) / (self.opened - self.closed), 0.0), 1.0)
            return values
        except Exception as exc:
            self.errors += 1
            if self.errors == 1 or self.errors % 30 == 0:
                logger.warning("ReBot read failed (%s); %d so far", exc, self.errors)
            return None

    def command(self, values):
        values = clamp(values)
        values[6] = self.closed + values[6] * (self.opened - self.closed)
        for motor, value in zip(self.motors, values, strict=True):
            motor.send_pos_vel(pos=value, vlim=self.speed)

    def switch_off(self):
        self.ctrl.disable_all()  # called only after the verified walk to rest
        self.energized = False

    def close(self):
        if self.ctrl is not None:
            # close_bus/close release resources; shutdown would also disable motors.
            try:
                self.ctrl.close_bus()
            finally:
                try:
                    for motor in self.motors:
                        motor.close()
                finally:
                    self.ctrl.close()
                    self.ctrl = None
                    self.motors = []


class Follower(Followers):
    """Reuse the verified park/home walk, with one arm and ReBot units/limits."""

    park_speed_env = "REBOT_PARK_DEG_S"

    def __init__(self, arm, cameras, fields, max_step):
        super().__init__({"right": arm}, cameras, StepLimiter(max_step))
        self.fields = fields
        self.parked = True

    def _open_arms(self, energize):
        # Unlike the Metal pair, there is no other arm to roll back on failure.
        # Arm.connect closes the bus without disabling a partially enabled arm.
        self.arms["right"].connect(energize)

    def connect(self):
        try:
            super().connect()
            self.parked = False
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self):
        try:
            for camera in self.cameras.values():
                try:
                    if camera._cam.is_connected or camera._cam.thread is not None:
                        camera._cam.disconnect()
                except Exception as exc:
                    logger.warning("camera cleanup: %s", exc)
        finally:
            self.arms["right"].close()

    def get_observation(self):
        reading = self.arms["right"].read()
        state = dict(zip(self.fields, reading)) if reading is not None else {}
        self.limiter.present = dict(state)  # a failed read must also prevent writes
        frames = {name: frame for name, cam in self.cameras.items() if (frame := cam.read()) is not None}
        return state, frames

    def send_action(self, action):
        if self.parked or not self.limiter.present:
            return
        if set(action) != set(self.fields):
            raise ValueError("ReBot actions must contain exactly the configured arm fields")
        bounded = dict(zip(self.fields, clamp([action[name] for name in self.fields])))
        target = self.limiter.step(bounded)
        self.arms["right"].command([target[name] for name in self.fields])


def build_follower(cfg, fps):
    from lerobot.cameras import Cv2Backends
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    fields = fields_from_config(cfg)
    arm = Arm(required_env("REBOT_CAN"), env_str("REBOT_TRANSPORT", "dm-serial"),
              float(required_env("REBOT_GRIPPER_CLOSED_RAD")), float(required_env("REBOT_GRIPPER_OPEN_RAD")))
    max_step = env_float("REBOT_MAX_STEP_RAD", 0.03)
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("REBOT_MAX_STEP_RAD must be positive and finite")
    cameras = {}
    for name in cfg.video_tracks:
        key = f"REBOT_CAM_{name.removesuffix('_camera').upper()}"
        config = OpenCVCameraConfig(
            index_or_path=env_camera_id(key, required_env(key)), fps=fps,
            width=env_int("REBOT_CAM_WIDTH", 640), height=env_int("REBOT_CAM_HEIGHT", 480), fourcc="MJPG",
            backend=Cv2Backends.V4L2 if platform.system() == "Linux" else Cv2Backends.ANY,
        )
        cameras[name] = Camera(name, OpenCVCamera(config))
    return Follower(arm, cameras, fields, max_step)
