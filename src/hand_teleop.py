"""Teleop from a laptop webcam: each hand drives one Metal arm while it holds a gripper shape.

    uv run --group hands hand-teleop --offline   # webcam and IK only: try it at a desk, no robot
    uv run --group hands hand-teleop             # in the robot's room; `c` takes control

MediaPipe finds up to two hands in the raw webcam image; the operator's right hand drives the right
arm. A hand holds its arm while middle, ring and pinky are curled all the way into the palm, leaving
thumb and index as a gripper, and the gap between their tips sets the gripper. Lifting any of the
three out of the palm lets go: nothing more is sent for that arm, and the robot holds it. Every grip
anchors anew, so the arm follows the hand's motion and turn since the grip, never where the hand is
or how it is held: let go, move back, grip again, as with a mouse. As with a mouse, slow moves count
for less (`HAND_FINE_SCALE`), so careful ones are fine and a still hand holds the arm still.
dream_ik turns each target tool pose into joints. See tutorial/11-hand-teleop.md.

Keys: `c` takes or releases control, `p` parks or powers up the arms, `x` quits. Saying "hand teleop" to the voice
agent (tutorial/10-voice-tasks.md) hands it control too, through `uv run dream-1-tasks`.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.common import (portal_config, HardwareThread, claim_or_release, env_camera_id, env_float, env_int, env_str,
                          hand_teleop_identity, load_env, mint_token, pace, request_park, required_env)
from utils.hands import (BONES, INDEX_TIP, THUMB_TIP, ArmFollower, ArmsIK, Clutch, Hand, HandPose, HandSize, Mapping,
                         assign_sides, camera_to_base, hand_pose, turn_between)
from utils.hotkeys import Hotkeys, input_backend_warning
from utils.metal import SIDES, arm_fields, clamp_to_limits
from utils.rerun_view import HAND_TELEOP_VIEWER_PORT, Preview

WEBCAM_VIEW = "webcam"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
             "hand_landmarker.task")
STALE_S = 0.25  # hands not seen for this long are let go
OBS_STALE_S = 0.5  # nothing is sent on a robot view older than this
MAX_JUMP_M = 0.08  # a palm moving this far between two frames is a tracking glitch (or a hand swap): let go
MAX_FLIP_RAD = math.radians(45)  # a palm turning this far between two frames is one too, when the gripper turns


def model_path() -> str:
    """MediaPipe's hand landmarker: `HAND_MODEL_PATH`, or Google's published model, downloaded once."""
    if path := env_str("HAND_MODEL_PATH", ""):
        return path
    cached = Path.home() / ".cache" / "portal-makermods" / "hand_landmarker.task"
    if not cached.exists():
        print(f"[hands] downloading MediaPipe's hand model to {cached}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        partial = cached.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(cached)
    return str(cached)


@dataclass(frozen=True)
class Sighting:
    """One webcam frame and the hands in it."""

    seq: int
    at: float  # s, monotonic, when the frame was read
    image: np.ndarray  # BGR, raw (unmirrored)
    hands: list[Hand]


class Webcam:
    """The webcam and MediaPipe on their own thread; `latest()` is the newest frame's hands.

    `source` is a camera index (a /dev/videoN path on Linux) or a video file, played in real time.
    """

    def __init__(self, source: int | str, width: int, height: int) -> None:
        self.source, self.width, self.height = source, width, height
        self.frames, self.errors = 0, 0
        self._latest: Optional[Sighting] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture = self._landmarker = None
        self._file_fps = 0.0

    def start(self) -> None:
        import mediapipe as mp  # lazy: the `hands` dependency group
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path()),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        # Opened here, on the main thread: macOS asks for camera access from it.
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"webcam {self.source!r} did not open (HAND_CAMERA; on macOS, allow camera access "
                               "for this terminal under System Settings > Privacy & Security > Camera)")
        if isinstance(self.source, str) and os.path.isfile(self.source):
            self._file_fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._thread = threading.Thread(target=self._run, name="webcam", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        started = last_ms = None
        seq = 0
        while not self._stop.is_set():
            ok, bgr = self._capture.read()
            now = time.monotonic()
            if not ok:
                self.errors += 1
                time.sleep(0.05)
                continue
            if self._file_fps:  # a recording plays at its own pace
                started = started or now
                time.sleep(max(0.0, started + seq / self._file_fps - now))
                now = time.monotonic()
            started = started or now
            ms = max(int((now - started) * 1000), (last_ms or -1) + 1)  # VIDEO mode needs rising timestamps
            last_ms = ms
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result = self._landmarker.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                                      data=rgb), ms)
            h, w = bgr.shape[:2]
            hands = [Hand(label=handed[0].category_name.lower(),
                          image=np.array([[p.x * w, p.y * h] for p in image]),
                          world=np.array([[p.x, p.y, p.z] for p in world]))
                     for handed, image, world in zip(result.handedness, result.hand_landmarks,
                                                     result.hand_world_landmarks)]
            seq += 1
            self.frames += 1
            with self._lock:
                self._latest = Sighting(seq, now, bgr, hands)

    def latest(self) -> Optional[Sighting]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
        if self._landmarker is not None:
            self._landmarker.close()


@dataclass
class Arm:
    """One arm's hand: the clutch, the follower, and the latest sighting of the hand."""

    clutch: Clutch
    follower: ArmFollower
    hand: Optional[HandPose] = None
    image: Optional[np.ndarray] = None  # (21, 2) pixels, for the overlay
    last_position: Optional[np.ndarray] = None
    last_rotation: Optional[np.ndarray] = None
    size: HandSize = field(default_factory=HandSize)  # the palm's size, steadier than any one frame's
    grip_size: Optional[float] = None  # held for the length of a grip, so depth cannot drift with it
    events: list[str] = field(default_factory=list)


class Pilot:
    """Hand sightings in, Portal actions out. `see` takes each webcam frame, `tick` each control tick.

    `q` and `grip` are what this operator holds each arm at: None until it first has control and a
    fresh state, and again after control changes hands, so it always starts from where the arms are.
    """

    def __init__(self, ik: ArmsIK, mappings: dict[str, Mapping], fps: float, focal_px: float, swap: bool) -> None:
        self.ik, self.fps, self.focal_px, self.swap = ik, fps, focal_px, swap
        self.arms = {side: Arm(Clutch(), ArmFollower(mappings[side])) for side in SIDES}
        self.q: Optional[dict[str, np.ndarray]] = None
        self.grip: Optional[dict[str, float]] = None
        self.seen = 0
        self.ik_ms = 0.0

    def see(self, sighting: Optional[Sighting], now: float) -> None:
        """Read the newest webcam frame; each frame once. No frame for STALE_S lets go of every arm."""
        if sighting is not None and sighting.seq == self.seen and now - sighting.at < STALE_S:
            return
        fresh = sighting is not None and now - sighting.at < STALE_S
        hands = assign_sides(sighting.hands, self.swap) if fresh else {}
        if fresh:
            self.seen = sighting.seq
            h, w = sighting.image.shape[:2]
        for side, arm in self.arms.items():
            hand = hands.get(side)
            if hand is not None:
                arm.size.add(hand.world)
            size = arm.grip_size if arm.follower.engaged else arm.size.value
            pose = hand_pose(hand, sighting.at, self.focal_px, (w / 2, h / 2), size) if hand is not None else None
            jumped = (pose is not None and arm.last_position is not None
                      and np.linalg.norm(pose.position - arm.last_position) > MAX_JUMP_M)
            flipped = (pose is not None and arm.last_rotation is not None and arm.follower.mapping.rotate != "off"
                       and turn_between(arm.last_rotation, pose.rotation) > MAX_FLIP_RAD)
            arm.last_position = None if pose is None else pose.position
            arm.last_rotation = None if pose is None else pose.rotation
            arm.hand, arm.image = pose, None if hand is None else hand.image
            if (jumped or flipped) and arm.clutch.closed:
                arm.events.append(f"the hand {'jumped' if jumped else 'flipped'} (tracking glitch?): let go")
            arm.clutch.update(None if pose is None or jumped or flipped else pose.curl)
            if pose is None or jumped or flipped:
                arm.follower.forget()
            else:
                arm.follower.observe(pose, moving=arm.clutch.closed)

    def take_over(self, state: dict[str, float]) -> None:
        self.q = {side: np.array([state[name] for name in arm_fields(side)[:6]]) for side in SIDES}
        self.grip = {side: float(state[arm_fields(side)[6]]) for side in SIDES}

    def drop(self) -> None:
        """Control changed hands: forget the held targets; the next grip starts from the arms' state."""
        self.q = self.grip = None
        for arm in self.arms.values():
            arm.follower.release()

    def tick(self, allowed: bool, state: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
        """The action to send this tick, or None to send nothing (the robot holds).

        `allowed`: this operator has control and a fresh view. `state`: the robot's latest state,
        needed once to start from where the arms are.
        """
        if not allowed:
            for side, arm in self.arms.items():
                if arm.follower.engaged:
                    arm.follower.release()
                    arm.events.append("let go: no control or no fresh robot view")
            return None
        gripping = [side for side, arm in self.arms.items() if arm.clutch.closed and arm.hand is not None]
        if self.q is None:
            if not gripping or state is None or any(name not in state for side in SIDES for name in arm_fields(side)):
                return None
            self.take_over(state)
        for side, arm in self.arms.items():
            if side in gripping and not arm.follower.engaged:
                arm.grip_size = arm.size.value
                arm.follower.engage(self.ik.fk(self.q[side], side), self.grip[side])
                arm.events.append("following" + (" (gripper waits for the pinch to reach it)"
                                                 if arm.follower.grip_waiting else ""))
            elif side not in gripping and arm.follower.engaged:
                arm.follower.release()
                arm.events.append("let go: holding")
        engaged = [side for side, arm in self.arms.items() if arm.follower.engaged]
        if not engaged:
            return None

        goals = {side: self.arms[side].follower.goal(self.fps) if side in engaged else self.ik.fk(self.q[side], side)
                 for side in SIDES}
        started = time.perf_counter()
        command, reachable = self.ik.step(goals, self.q)
        self.ik_ms = (time.perf_counter() - started) * 1e3
        for side in engaged:
            follower = self.arms[side].follower
            follower.settle(reachable[side])
            self.q[side], self.grip[side] = command[side], follower.grip_mm
        action = {}
        for side in SIDES:
            action.update(zip(arm_fields(side), [*map(float, self.q[side]), self.grip[side]]))
        return clamp_to_limits(action)

    def drain_events(self) -> list[str]:
        out = []
        for side, arm in self.arms.items():
            out += [f"{side} arm: {event}" for event in arm.events]
            arm.events.clear()
        return out


def caption(image: np.ndarray, text: str, x: float, bottom: float, scale: float = 0.5, centred: bool = True) -> None:
    """White text on a dark box centred on `x` (or starting there), its bottom at `bottom`, kept inside the image.

    One stroke weight: OpenCV 5 draws a thicker stroke with wider letters, so a thick outline under thin text
    drifts apart along the line.
    """
    font, pad = cv2.FONT_HERSHEY_SIMPLEX, 3
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    h, w = image.shape[:2]
    left = int(np.clip(x - tw / 2 if centred else x, pad + 1, max(pad + 1, w - tw - pad - 1)))
    bottom = int(np.clip(bottom, th + base + 2 * pad, h - 1))
    top = bottom - th - base - 2 * pad
    cv2.rectangle(image, (left - pad, top), (left + tw + pad, bottom), (20, 20, 20), cv2.FILLED)
    cv2.putText(image, text, (left, bottom - base - pad), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw(sighting: Sighting, pilot: Pilot, status: str) -> np.ndarray:
    """The webcam as a mirror, each hand's skeleton in its arm's state, and a status line. RGB."""
    image = sighting.image.copy()
    colours = {"engaged": (80, 220, 80), "seen": (40, 200, 240)}  # BGR
    for side, arm in pilot.arms.items():
        if arm.image is None:
            continue
        colour = colours["engaged" if arm.follower.engaged else "seen"]
        points = [tuple(int(v) for v in p) for p in arm.image]
        for a, b in BONES:
            cv2.line(image, points[a], points[b], colour, 2, cv2.LINE_AA)
        if arm.follower.engaged:
            cv2.line(image, points[THUMB_TIP], points[INDEX_TIP], (60, 60, 255), 4, cv2.LINE_AA)
    image = cv2.flip(image, 1)  # draw text after the flip, so it reads
    h, w = image.shape[:2]
    for side, arm in pilot.arms.items():
        if arm.image is None:
            continue
        x, y = w - 1 - arm.image[:, 0].mean(), arm.image[:, 1].min() - 8  # centred just above the hand
        if arm.follower.engaged:
            d = arm.follower.moved() * 1000
            label = f"{side.upper()} {d[0]:+.0f} fwd {-d[1]:+.0f} right {d[2]:+.0f} up mm"
            if arm.follower.mapping.rotate != "off":
                label += f", turn {math.degrees(arm.follower.turned()):.0f} deg"
            label += f", grip {arm.follower.grip_mm:.0f}"
            label += " (pinch to take the gripper)" if arm.follower.grip_waiting else ""
        else:
            label = f"{side.upper()} curl {' '.join(f'{c:.1f}' for c in arm.hand.curl)}" if arm.hand else side.upper()
        caption(image, label, x, y)
    caption(image, status, 4, h - 4, scale=0.55, centred=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_pilot(fps: float, ik: ArmsIK, rotate: str, frame_width: int) -> Pilot:
    hfov = math.radians(env_float("HAND_CAMERA_HFOV_DEG", 70.0))
    mappings = {side: Mapping(to_base=camera_to_base(env_float(f"HAND_{side.upper()}_YAW_DEG", 0.0)),
                              scale=env_float("HAND_SCALE", 1.0), fine_scale=env_float("HAND_FINE_SCALE", 0.4),
                              rotate=rotate,
                              turn_deadband_rad=math.radians(env_float("HAND_TURN_DEADBAND_DEG", 5.0)),
                              max_speed_m_s=env_float("HAND_MAX_SPEED_M_S", 0.5))
                for side in SIDES}
    return Pilot(ik, mappings, fps, focal_px=frame_width / 2 / math.tan(hfov / 2),
                 swap=env_int("HAND_SWAP_SIDES", 0) != 0)


async def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="no robot: the arms start at the IK's rest pose and nothing is sent")
    parser.add_argument("--rotate", choices=("full", "yaw", "off"), default=env_str("HAND_ROTATE", "full"),
                        help="what the hand's turning does to the gripper (default: full, it turns as the hand "
                             "does; yaw: about vertical only; off: it keeps its orientation)")
    args = parser.parse_args()

    if args.offline:
        cfg, url, room, fps, cameras = None, "", "offline", env_int("PORTAL_FPS", 30), []
    else:
        from livekit.portal import OperatorConfig

        url, room = required_env("LIVEKIT_URL"), required_env("LIVEKIT_ROOM")
        cfg = portal_config(OperatorConfig, room, "metal")
        fps, cameras = env_int("PORTAL_FPS", cfg.fps), list(cfg.video_tracks)

    width, height = env_int("HAND_CAMERA_WIDTH", 640), env_int("HAND_CAMERA_HEIGHT", 480)
    worker = HardwareThread("hand-ik")  # the IK off the event loop
    ik = ArmsIK(fps, env_float("HAND_TABLE_Z_M", 0.0), env_float("HAND_MAX_JOINT_RAD_S", 1.5))
    pilot = build_pilot(fps, ik, args.rotate, width)
    webcam = Webcam(env_camera_id("HAND_CAMERA", "0"), width, height)
    hotkeys = Hotkeys(global_tap=env_int("PORTAL_HOTKEYS_GLOBAL", 1) != 0)
    rest = {name: value for side in SIDES  # offline: both arms at the IK's rest, grippers half open
            for name, value in zip(arm_fields(side), [*ik.rest, 50.0])}

    op, preview = None, None
    latest_obs, obs_at, shown_ts = None, 0.0, None
    control_changed = False
    try:
        print("[hands] compiling the IK (a few seconds the first time) ...")
        await worker.run(ik.warmup)
        webcam.start()
        preview = Preview(room, cameras + [WEBCAM_VIEW], port=HAND_TELEOP_VIEWER_PORT)
        if not args.offline:
            from livekit.portal import Operator, frame_bytes_to_numpy_rgb

            op = Operator(cfg)

            def on_observation(obs) -> None:
                nonlocal latest_obs, obs_at
                latest_obs, obs_at = obs, time.monotonic()

            def on_active_operator_changed(identity) -> None:
                nonlocal control_changed
                control_changed = True
                print(f"[hands] active operator: {identity}")

            op.on_observation(on_observation)
            op.on_active_operator_changed(on_active_operator_changed)
            identity = hand_teleop_identity()
            print(f"[hands] connecting to {url} as '{identity}' in room '{room}' ...")
            await op.connect(url, mint_token(identity, room))

        hotkeys.start()
        if (why := input_backend_warning()) is not None:
            print(f"[hands] global hotkeys will not fire ({why}); press them in this terminal")
        print("[hands] curl middle, ring and pinky into the palm to grip an arm; thumb to index sets its gripper; "
              "lift one out to let go" + ("" if args.rotate == "off" else "; turning the hand turns the gripper"))
        print("[hands] " + ("offline: nothing is sent; x quits" if args.offline else
                            "c takes or releases control, p parks, x quits"))
        shown_seq = 0
        async for tick in pace(fps):
            keys = hotkeys.drain()
            if "x" in keys:
                break
            if op is not None:
                if "c" in keys:
                    await claim_or_release(op, "hands")
                if "p" in keys:
                    await request_park(op, "hands")

            now = time.monotonic()
            sighting = webcam.latest()
            pilot.see(sighting, now)

            if op is None:
                mine, obs, state = True, None, rest
            else:
                if control_changed:
                    control_changed = False
                    pilot.drop()
                obs = latest_obs
                fresh = obs is not None and now - obs_at < OBS_STALE_S
                mine = op.active_operator() == op.local_identity()
                state = obs.state if fresh else None
                mine = mine and fresh
                if obs is not None and obs.timestamp_us != shown_ts:
                    shown_ts = obs.timestamp_us
                    for name, frame in obs.frames.items():
                        preview.show(name, frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height))

            action = await worker.run(pilot.tick, mine, state)
            for line in pilot.drain_events():
                print(f"[hands] {line}")
            if action is not None and op is not None:
                op.send_action(action, in_reply_to_ts_us=obs.timestamp_us)

            if sighting is not None and sighting.seq != shown_seq:
                shown_seq = sighting.seq
                status = ("offline" if op is None else "control: yes" if mine else
                          "control: no (press c)" if op.active_operator() != op.local_identity() else
                          "control: yes, no fresh robot view")
                preview.show(WEBCAM_VIEW, draw(sighting, pilot, status))

            if tick and tick % (fps * 5) == 0:
                held = [side for side, arm in pilot.arms.items() if arm.follower.engaged] or ["none"]
                print(f"[hands] webcam {webcam.frames / 5:.0f} fps, IK {pilot.ik_ms:.1f} ms, following: "
                      + ", ".join(held))
                webcam.frames = 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[hands] stopping")
    finally:
        hotkeys.stop()
        webcam.stop()
        if preview is not None:
            preview.close()
        worker.close()
        if op is not None:
            try:
                await op.disconnect()
            finally:
                op.close()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
