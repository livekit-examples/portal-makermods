"""Webcam hands as arm targets: the gesture clutch, the hand's pose, and the relative mapping.

Numpy on MediaPipe's 21 hand landmarks, so all of it tests without a camera or a model; the webcam
loop is `src/hand_teleop.py`. See tutorial/11-hand-teleop.md.

Two frames:
- camera: OpenCV's, of the raw (unmirrored) webcam image: x right, y down, z away from the lens.
  MediaPipe's world landmarks use these axes, in metres, about the hand's centre, but turned as the
  hand looks from the lens rather than from straight ahead (see `view_rotation`).
- base: each arm's own, as dream_ik's FK has it: x forward, y to the robot's left, z up.

By default motion is in the gripper's POV: toward the lens advances along its fingers. Rotation
uses the palm's own axes, so twisting the wrist rolls the gripper regardless of how the hand is
held. The optional base frame retains the original standing-behind-the-arms mapping.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from utils.metal import GRIPPER_MAX_MM, SIDES

HAND_PROFILES = ("metal", "rebot", "widowx", "yam")

# MediaPipe's landmark indices.
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP = 0, 4, 5, 8
MIDDLE_MCP, MIDDLE_TIP, RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP = 9, 12, 13, 16, 17, 20
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)  # stays put while the fingers move
CLOSED_FINGERS = ((MIDDLE_MCP, MIDDLE_TIP), (RING_MCP, RING_TIP), (PINKY_MCP, PINKY_TIP))
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))

# Camera axes into base axes: toward the lens (-z) is forward (+x), the image's right (the operator's
# left) is the robot's left (+y), and the image's up (-y) is up (+z).
CAMERA_TO_BASE = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# Dream's TCP: z along the fingers, x across the jaws, y = z cross x. Looking out along z,
# call +x right and -y up. Lateral and vertical controls are reversed from the initial mapping:
# raw +x maps to tool +x, raw -y (hand up) maps to tool +y. Wrist turns use HAND_TO_TOOL separately.
CAMERA_TO_TOOL = np.diag([1.0, -1.0, -1.0])
# Palm x (wrist toward knuckles) -> approach/roll z; palm y -> jaw x; palm normal z -> tool y.
HAND_TO_TOOL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

# Tip-to-wrist over knuckle-to-wrist, per finger: about 1.8 straight. Below 1.0 the tip has folded back
# past its knuckle, into the palm; a hooked or half-curled finger stays above it.
CURLED_BELOW = 1.0
OPEN_ABOVE = 1.15


@dataclass(frozen=True)
class Hand:
    """One hand in one webcam frame, as MediaPipe reads the raw image."""

    label: str  # "left" or "right": the operator's own hand
    image: np.ndarray  # (21, 2) pixels
    world: np.ndarray  # (21, 3) m, camera axes, about the hand's centre


@dataclass(frozen=True)
class HandPose:
    """What one sighting says about a hand: where it is, how it is turned, how far the pinch is open."""

    at: float  # s, monotonic
    position: np.ndarray  # (3,) m, the palm's centre in the camera frame
    rotation: np.ndarray  # (3, 3) the palm's axes in the camera frame
    pinch: float  # m, thumb tip to index tip
    curl: np.ndarray  # (3,) middle, ring, pinky (see `curl`)


def curl(world: np.ndarray) -> np.ndarray:
    """Middle, ring and pinky: each tip's distance from the wrist over its knuckle's. Lower is more curled."""
    wrist = world[WRIST]
    return np.array([np.linalg.norm(world[tip] - wrist) / max(np.linalg.norm(world[mcp] - wrist), 1e-6)
                     for mcp, tip in CLOSED_FINGERS])


def pinch(world: np.ndarray) -> float:
    return float(np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP]))


def palm_size(world: np.ndarray) -> float:
    """How big the palm is (m): its five rigid points' RMS distance from their centre."""
    palm = world[list(PALM)]
    return float(np.sqrt(((palm - palm.mean(axis=0)) ** 2).sum(axis=1).mean()))


def palm_position(image: np.ndarray, world: np.ndarray, focal_px: float, center_px: Sequence[float],
                  size_m: Optional[float] = None) -> Optional[np.ndarray]:
    """The palm's centre in the camera frame (m), or None for a degenerate hand.

    The palm's five rigid points (wrist and knuckles), shaped and turned as the world landmarks have
    them, are placed where they project onto their pixels (a least-squares perspective fit), so depth
    comes from the palm's size in the image. The fingers do not count: a pinch never moves the arm.
    `size_m` sets the palm's size: MediaPipe estimates its metric scale afresh every frame, and a few
    percent of wander there is centimetres of depth at arm's length, so `HandSize` gives a steadier one.
    """
    palm = world[list(PALM)] - world[list(PALM)].mean(axis=0)
    if size_m is not None:
        palm = palm * (size_m / max(palm_size(world), 1e-9))
    pixels = image[list(PALM)]
    model = palm @ view_rotation(pixels.mean(axis=0), focal_px, center_px).T  # camera axes
    u, v = (pixels - np.asarray(center_px)).T
    # f (X + mx) = u (Z + mz) and f (Y + my) = v (Z + mz), for the centre (X, Y, Z): linear in it.
    a = np.zeros((2 * len(palm), 3))
    a[0::2, 0], a[0::2, 2], a[1::2, 1], a[1::2, 2] = focal_px, -u, focal_px, -v
    b = np.stack([u * model[:, 2] - focal_px * model[:, 0], v * model[:, 2] - focal_px * model[:, 1]], axis=1).ravel()
    centre = np.linalg.lstsq(a, b, rcond=None)[0]
    return centre if centre[2] > 0.05 else None


def palm_rotation(world: np.ndarray) -> np.ndarray:
    """The palm's axes in the camera frame: x from the wrist to the knuckles' middle (steadier than to
    any one knuckle), z the palm's normal, y = z × x."""
    x = world[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0) - world[WRIST]
    x = x / max(np.linalg.norm(x), 1e-9)
    z = np.cross(x, world[INDEX_MCP] - world[PINKY_MCP])
    z = z / max(np.linalg.norm(z), 1e-9)
    return np.stack([x, np.cross(z, x), z], axis=1)


class HandSize:
    """The palm's size (m) as the median of the last few seconds of sightings: MediaPipe's metric scale
    can wander from frame to frame, a hand does not."""

    def __init__(self, keep: int = 150) -> None:
        self._sizes: deque[float] = deque(maxlen=keep)

    def add(self, world: np.ndarray) -> None:
        self._sizes.append(palm_size(world))

    @property
    def value(self) -> Optional[float]:
        return float(np.median(self._sizes)) if self._sizes else None


def view_rotation(pixel: Sequence[float], focal_px: float, center_px: Sequence[float]) -> np.ndarray:
    """The turn that points the lens's axis along the ray through `pixel`.

    MediaPipe reads a hand's world landmarks off a crop around it, so they come turned as the hand
    looks along that ray. Turning them by this puts them in the camera's axes; without it, a hand slid
    toward the image's edge would read as turned by up to half the field of view.
    """
    ray = np.array([(pixel[0] - center_px[0]) / focal_px, (pixel[1] - center_px[1]) / focal_px, 1.0])
    axis = np.array([-ray[1], ray[0], 0.0])  # the lens's axis (0, 0, 1) × ray
    sin = float(np.linalg.norm(axis))
    if sin < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(axis / sin * math.atan2(sin, ray[2])).as_matrix()


def turn_between(a: np.ndarray, b: np.ndarray) -> float:
    """The angle (rad) of the turn from rotation `a` to rotation `b`."""
    return float(math.acos(np.clip((np.trace(b @ a.T) - 1.0) / 2.0, -1.0, 1.0)))


def hand_pose(hand: Hand, at: float, focal_px: float, center_px: Sequence[float],
              size_m: Optional[float] = None) -> Optional[HandPose]:
    position = palm_position(hand.image, hand.world, focal_px, center_px, size_m)
    if position is None:
        return None
    view = view_rotation(hand.image[list(PALM)].mean(axis=0), focal_px, center_px)
    return HandPose(at, position, view @ palm_rotation(hand.world), pinch(hand.world), curl(hand.world))


def assign_sides(hands: Sequence[Hand], swap: bool = False) -> dict[str, Hand]:
    """Which arm each hand drives: the operator's right hand drives the right arm.

    The model's label decides, unless both hands carry the same one; then position does, the
    operator's right hand being on the raw image's left. `swap` is for a webcam that mirrors.
    """
    if swap:
        hands = [Hand({"left": "right", "right": "left"}.get(h.label, h.label), h.image, h.world) for h in hands]
    if len(hands) == 2 and hands[0].label == hands[1].label:
        on_left, on_right = sorted(hands, key=lambda h: h.image[WRIST, 0])
        return {"right": on_left, "left": on_right}
    out: dict[str, Hand] = {}
    for hand in hands:
        if hand.label in SIDES:
            out.setdefault(hand.label, hand)
    return out


def camera_to_base(yaw_deg: float = 0.0) -> np.ndarray:
    """Camera axes into one arm's base axes. `yaw_deg` is how far that arm's base x is turned to the
    left of straight ahead (counterclockwise, seen from above): 0 for an arm facing the workspace."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T @ CAMERA_TO_BASE


class Clutch:
    """Whether a hand holds its arm: the three fingers curled all the way into the palm for a few frames
    in a row closes it, any of them coming back out (or the hand gone) for a frame or two opens it. The
    gap between the two thresholds keeps a finger near either one from chattering."""

    def __init__(self, close_frames: int = 3, open_frames: int = 2,
                 curled_below: float = CURLED_BELOW, open_above: float = OPEN_ABOVE) -> None:
        self.close_frames, self.open_frames = close_frames, open_frames
        self.curled_below, self.open_above = curled_below, open_above
        self.closed = False
        self._run = 0

    def update(self, curls: Optional[np.ndarray]) -> bool:
        """One sighting's curl values (None: no hand); returns whether the clutch is closed."""
        if curls is None:
            self.closed, self._run = False, 0
            return False
        towards_change = (curls > self.open_above).any() if self.closed else (curls < self.curled_below).all()
        self._run = self._run + 1 if towards_change else 0
        if self._run >= (self.open_frames if self.closed else self.close_frames):
            self.closed, self._run = not self.closed, 0
        return self.closed


class OneEuro:
    """The 1€ filter (Casiez et al., 2012): smooth when the hand is still, little lag when it moves."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff_hz, beta, d_cutoff_hz
        self._x: Optional[np.ndarray] = None
        self._dx = None
        self._at = 0.0

    @staticmethod
    def _alpha(cutoff_hz: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, x, at: float) -> np.ndarray:
        self._x, self._dx, self._at = np.array(x, dtype=np.float64), np.zeros_like(np.asarray(x, np.float64)), at
        return self._x.copy()

    def clear(self) -> None:
        """Start afresh: the next value is taken as it is."""
        self._x = None

    @property
    def speed(self) -> float:
        """How fast the input moves (its units per s, smoothed)."""
        return 0.0 if self._x is None else float(np.linalg.norm(self._dx))

    def __call__(self, x, at: float) -> np.ndarray:
        if self._x is None or at <= self._at:
            return self.reset(x, at) if self._x is None else self._x.copy()
        dt = at - self._at
        x = np.asarray(x, dtype=np.float64)
        self._dx = self._dx + self._alpha(self.d_cutoff, dt) * ((x - self._x) / dt - self._dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        self._at = at
        return self._x.copy()


class OneEuroTurn:
    """The 1€ filter on rotations: each sighting pulls the estimate along the shortest turn toward it."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff_hz, beta, d_cutoff_hz
        self._r: Optional[np.ndarray] = None
        self._w = np.zeros(3)
        self._at = 0.0

    def clear(self) -> None:
        self._r = None

    def __call__(self, rotation: np.ndarray, at: float) -> np.ndarray:
        if self._r is None:
            self._r, self._w, self._at = np.array(rotation, dtype=np.float64), np.zeros(3), at
        elif at > self._at:
            dt = at - self._at
            towards = Rotation.from_matrix(rotation @ self._r.T).as_rotvec()
            self._w = self._w + OneEuro._alpha(self.d_cutoff, dt) * (towards / dt - self._w)
            cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._w))
            self._r = Rotation.from_rotvec(OneEuro._alpha(cutoff, dt) * towards).as_matrix() @ self._r
            self._at = at
        return self._r.copy()


def _twist_about_z(rotation: np.ndarray) -> float:
    """The part of a rotation that turns about z (swing-twist), in rad."""
    return 2.0 * math.atan2(rotation[1, 0] - rotation[0, 1], 1.0 + np.trace(rotation))


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Mapping:
    """How a hand's motion becomes a tool pose. Speeds bound the target, before the IK's joint caps."""

    to_base: np.ndarray  # (3, 3), from `camera_to_base`
    frame: str = "tool"  # "tool": gripper POV; "base": the original fixed camera-to-base mapping
    scale: float = 1.0  # tool metres per hand metre
    fine_scale: float = 0.4  # how much of `scale` a slow hand gets: careful moves (and a still hand's wobble) shrink
    fine_below_m_s: float = 0.04  # a hand this slow or slower moves the tool at `fine_scale`
    full_above_m_s: float = 0.2  # this fast or faster at the full `scale`; smoothly in between
    rotate: str = "full"  # "full": the tool turns as the hand does; "yaw": about vertical only; "off": it keeps its own
    turn_deadband_rad: float = math.radians(5)  # the hand's turn since the grip, less this; smaller turns do nothing
    max_hand_turn_rad: float = math.radians(90)  # past this from the grip, the hand's anchor turns along
    max_speed_m_s: float = 0.5
    max_turn_rad_s: float = math.pi
    pinch_closed_m: float = 0.03  # thumb to index tip at and below this: gripper closed
    pinch_open_m: float = 0.09  # at and above this: fully open
    smoothing_hz: float = 1.0  # the 1€ filters' cutoffs for a still hand: across the image (the pinch twice it)
    depth_smoothing_hz: float = 0.4  # toward the lens, where the estimate is several times noisier
    depth_beta: float = 2.0
    turn_smoothing_hz: float = 0.3  # and the hand's turn
    turn_beta: float = 0.5
    turn_d_cutoff_hz: float = 0.5
    hold_m: float = 0.001  # the target stays put until the hand's motion takes it this far (after the scales)
    hold_rad: float = math.radians(1.5)  # and its turn this far
    slip_m: float = 0.015  # the target this far from where the IK could go moves the anchor along
    slip_rad: float = math.radians(10)
    takeover_mm: float = 0.1 * GRIPPER_MAX_MM  # the pinch must come this close to the gripper to take it

    def __post_init__(self) -> None:
        if self.frame not in ("tool", "base"):
            raise ValueError("hand frame must be tool or base")

    def gain(self, speed_m_s: float) -> float:
        """How much of the hand's motion the tool makes at this hand speed, before `scale`."""
        t = float(np.clip((speed_m_s - self.fine_below_m_s) / max(self.full_above_m_s - self.fine_below_m_s, 1e-9),
                          0.0, 1.0))
        return self.fine_scale + (1.0 - self.fine_scale) * t * t * (3.0 - 2.0 * t)

    def grip_mm(self, pinch_m: float) -> float:
        span = max(self.pinch_open_m - self.pinch_closed_m, 1e-6)
        return float(np.clip((pinch_m - self.pinch_closed_m) / span, 0.0, 1.0)) * GRIPPER_MAX_MM


@dataclass
class ArmFollower:
    """One arm following one hand's motion since the clutch closed, never the hand's absolute place.

    Every sighting of the hand is filtered (`observe`), gripping or not, so a grip anchors to a settled
    estimate rather than to one frame's noise. `engage` anchors the hand to the tool. While engaged,
    each sighting moves the target by the hand's motion in the selected frame, scaled by how fast the hand
    goes: a slow, careful hand moves the tool at `fine_scale`, a quick one at full `scale`, like a
    mouse. A still hand holds the tool still: the target moves only once the hand has taken it
    `hold_m` from where it is (or turned it `hold_rad`), so jitter never reaches the arm. Each tick
    `goal` moves toward the target no faster than the speed limits. Where the arm cannot follow (the
    floor, its reach), `settle` moves the target to where the arm is, and the hand drives it from
    there at once instead of first having to come back.

    Tool-frame translation uses the latest commanded tool orientation, updated after IK's joint cap;
    turning the gripper redirects the next forward motion. Rotation is the palm's body-relative turn
    since engagement, mapped to the tool's axes: a twist around wrist-to-knuckles is always tool roll.
    In base mode turns instead use the camera axes. The first few degrees are dropped, so a hand that only moves keeps the
    tool steady. A turn the arm cannot make stops the tool where it can go, and the turn comes back to
    the hand's when the hand does: with `rotate` "full", returning to the hand's anchor returns to the
    tool's anchor (give or take the hold). Past `max_hand_turn_rad` the hand's anchor turns along with it: turning
    back acts at once, and turning farther takes a new grip, as moving farther does.

    The gripper takes over softly: it holds where it was until the pinch comes to it, so gripping
    again with an open hand never drops what the arm is holding.
    """

    mapping: Mapping
    engaged: bool = False
    grip_mm: float = 0.0
    _hand_pos: Optional[np.ndarray] = field(default=None, repr=False)  # filtered, camera frame
    _hand_rot: Optional[np.ndarray] = field(default=None, repr=False)  # filtered, camera frame
    _pinch: float = 0.0
    _anchor_hand_rot: np.ndarray = field(default=None, repr=False)
    _anchor_tcp: np.ndarray = field(default=None, repr=False)  # (4, 4)
    _target: np.ndarray = field(default=None, repr=False)  # (3,) where the hand's motion has put the tool
    _held: np.ndarray = field(default=None, repr=False)  # (3,) the target, moved only past `hold_m`
    _kept: np.ndarray = field(default=None, repr=False)  # (3, 3) the hand's turn past the deadband, held likewise
    _goal: np.ndarray = field(default=None, repr=False)  # (4, 4), the last target
    _turn: np.ndarray = field(default=None, repr=False)  # (3, 3), the last base-frame turn applied to the anchor
    _motion_rot: np.ndarray = field(default=None, repr=False)  # current commanded TCP orientation
    _grip_taken: bool = False
    _grip_side: float = 0.0

    def __post_init__(self) -> None:
        m = self.mapping
        self._across, self._depth = OneEuro(m.smoothing_hz, 4.0), OneEuro(m.depth_smoothing_hz, m.depth_beta)
        self._turning = OneEuroTurn(m.turn_smoothing_hz, m.turn_beta, m.turn_d_cutoff_hz)
        self._pinching = OneEuro(2 * m.smoothing_hz, 2.0)

    def observe(self, hand: HandPose, moving: bool = True) -> None:
        """Take one sighting of the hand; call once per webcam frame while it is seen. While engaged and
        `moving` (the clutch still closed), it moves the target and the turn."""
        last = self._hand_pos
        self._hand_pos = np.append(self._across(hand.position[:2], hand.at), self._depth(hand.position[2:], hand.at))
        self._hand_rot = self._turning(hand.rotation, hand.at)
        self._pinch = float(self._pinching([hand.pinch], hand.at)[0])
        if not (self.engaged and moving):
            return
        m = self.mapping
        if last is not None:
            gain = m.scale * m.gain(math.hypot(self._across.speed, self._depth.speed))
            axes = self._motion_rot @ CAMERA_TO_TOOL if m.frame == "tool" else m.to_base
            self._target = self._target + gain * (axes @ (self._hand_pos - last))
            if (gap := np.linalg.norm(off := self._target - self._held)) > m.hold_m:
                self._held = self._held + off * (1.0 - m.hold_m / gap)
        relative = (self._anchor_hand_rot.T @ self._hand_rot if m.frame == "tool"
                    else self._hand_rot @ self._anchor_hand_rot.T)
        turn = Rotation.from_matrix(relative).as_rotvec()
        if (angle := np.linalg.norm(turn)) > (cap := m.max_hand_turn_rad):
            beyond = turn * (1.0 - cap / angle)
            slip = Rotation.from_rotvec(beyond).as_matrix()
            self._anchor_hand_rot = (self._anchor_hand_rot @ slip if m.frame == "tool"
                                     else slip @ self._anchor_hand_rot)
            turn = turn - beyond
        angle = float(np.linalg.norm(turn))
        wanted = Rotation.from_rotvec(turn * (max(0.0, angle - m.turn_deadband_rad) / max(angle, 1e-9))).as_matrix()
        off = Rotation.from_matrix(wanted @ self._kept.T).as_rotvec()
        if (gap := np.linalg.norm(off)) > m.hold_rad:
            self._kept = Rotation.from_rotvec(off * (1.0 - m.hold_rad / gap)).as_matrix() @ self._kept

    def forget(self) -> None:
        """The hand is gone, or a sighting was a glitch: the next one starts the filters afresh."""
        self._across.clear(), self._depth.clear(), self._turning.clear(), self._pinching.clear()
        self._hand_pos = self._hand_rot = None

    def engage(self, tcp: np.ndarray, grip_mm: float) -> None:
        """Anchor the hand, as filtered so far, to the tool pose `tcp` (4, 4) and the gripper's opening."""
        assert self._hand_pos is not None, "observe the hand before engaging"
        self.engaged = True
        self._anchor_hand_rot, self._kept = self._hand_rot.copy(), np.eye(3)
        self._anchor_tcp, self._goal, self._turn = tcp.copy(), tcp.copy(), np.eye(3)
        self._motion_rot = tcp[:3, :3].copy()
        self._target, self._held = tcp[:3, 3].copy(), tcp[:3, 3].copy()
        self.grip_mm = grip_mm
        self._grip_side = math.copysign(1.0, self.mapping.grip_mm(self._pinch) - grip_mm)
        self._grip_taken = abs(self.mapping.grip_mm(self._pinch) - grip_mm) <= self.mapping.takeover_mm

    def release(self) -> None:
        self.engaged = False

    def goal(self, fps: float) -> np.ndarray:
        """This tick's target tool pose (4, 4) in the base frame; `grip_mm` is the gripper's."""
        m = self.mapping
        axes = self._anchor_tcp[:3, :3] @ HAND_TO_TOOL if m.frame == "tool" else m.to_base
        turn = axes @ self._kept @ axes.T
        self._turn = {"off": np.eye(3), "yaw": _rot_z(_twist_about_z(turn))}.get(m.rotate, turn)
        rotation = self._turn @ self._anchor_tcp[:3, :3]

        step = self._held - self._goal[:3, 3]
        if (n := np.linalg.norm(step)) > (limit := m.max_speed_m_s / fps):
            step *= limit / n
        swing = Rotation.from_matrix(rotation @ self._goal[:3, :3].T).as_rotvec()
        if (angle := np.linalg.norm(swing)) > (limit := m.max_turn_rad_s / fps):
            swing *= limit / angle
        self._goal = self._goal.copy()
        self._goal[:3, 3] += step
        self._goal[:3, :3] = Rotation.from_rotvec(swing).as_matrix() @ self._goal[:3, :3]
        self._motion_rot = self._goal[:3, :3].copy()  # standalone follower; settle replaces this after IK

        wanted = m.grip_mm(self._pinch)
        if not self._grip_taken and (abs(wanted - self.grip_mm) <= m.takeover_mm
                                     or math.copysign(1.0, wanted - self.grip_mm) != self._grip_side):
            self._grip_taken = True
        if self._grip_taken:
            self.grip_mm = wanted
        return self._goal.copy()

    @property
    def grip_waiting(self) -> bool:
        """The gripper is holding until the pinch comes to it."""
        return self.engaged and not self._grip_taken

    def settle(self, reachable: np.ndarray, commanded: Optional[np.ndarray] = None) -> None:
        """Where the IK says the arm can go this tick (4, 4). A place past it moves the target there; a
        turn past it holds the target there, and with `rotate` other than "full" turns the anchor too.
        `commanded` is FK after the joint speed cap, used for the next tool-frame movement's direction.
        """
        m = self.mapping
        self._motion_rot = (commanded if commanded is not None else reachable)[:3, :3].copy()
        miss = reachable[:3, 3] - self._goal[:3, 3]
        if np.linalg.norm(miss) > m.slip_m:
            self._anchor_tcp[:3, 3] += miss
            self._target, self._held = self._target + miss, self._held + miss
            self._goal[:3, 3] += miss
        off = Rotation.from_matrix(reachable[:3, :3] @ self._goal[:3, :3].T).magnitude()
        if off > m.slip_rad:
            if m.rotate != "full":  # some of the orientation is the anchor's alone: keep the one the arm has
                self._anchor_tcp[:3, :3] = self._turn.T @ reachable[:3, :3]
            self._goal[:3, :3] = reachable[:3, :3]  # full: the hand turns it back; only the target waits

    def moved(self) -> np.ndarray:
        """The target's offset from the anchor, base frame, m."""
        return self._goal[:3, 3] - self._anchor_tcp[:3, 3] if self.engaged else np.zeros(3)

    def turned(self) -> float:
        """How far the target has turned from the anchor, rad."""
        return turn_between(self._anchor_tcp[:3, :3], self._goal[:3, :3]) if self.engaged else 0.0


class ArmsIK:
    """The selected rig's IK one tick at a time: dream_ik's ChunkIK on the tool point between the fingertips.

    `table_z_m` is a floor under each arm, in its base frame (0 is the base's underside, so the table
    it stands on): targets are clipped above it and the solve keeps the arm's centreline off it. There
    is no arm-to-arm guard: the bases' spacing is not known here.

    Place comes before turn. Where a pose cannot be reached, the solve trades place against turn by
    their weights. dream_ik's own, 50 to 10 (Dream Labs' teleop), let a hand tilted past what the arm
    can do from its rest pose drag the tool 3 to 11 cm; at `TURN_WEIGHT` the turn gives way instead and
    the tool stays within a few millimetres, at the cost of ending reachable turns a degree or two short.
    """

    TURN_WEIGHT = 4.0

    def __init__(self, fps: float, table_z_m: float, max_joint_rad_s: float, profile: str = "metal", cfg=None) -> None:
        import jax.numpy as jnp  # lazy: JAX
        from dream_ik.solver import ChunkIK
        from dream_ik.specs import EMBODIMENTS, Embodiment, Region

        if profile not in HAND_PROFILES:
            raise ValueError(f"unsupported hand-teleop profile: {profile}")
        self.profile = profile
        self.sides = ("right",) if profile == "rebot" else SIDES
        model = "metal"
        if profile == "rebot":
            # Use Dream's built-in DM chain/tool point, with only the physical right arm.
            model = "portal_rebot_right"
            EMBODIMENTS.setdefault(model, Embodiment(arm=EMBODIMENTS["seeed_faas"].side_arm("right"),
                                                   sides=self.sides, obstacles={"right": ()}))
        elif profile in ("widowx", "yam"):
            from dream import arm_file
            from utils.common import ROOT
            model = arm_file.register(str(ROOT / "assets" / profile / "dream_arm.json"))

        from livekit.portal import OperatorConfig
        from utils.common import portal_config
        from utils.metal import arm_fields, LIMITS_RAD
        from utils.rebot import LIMITS as REBOT_LIMITS

        cfg = cfg if cfg is not None else portal_config(OperatorConfig, "offline", profile)
        self.fields, self.limits = {}, {}
        for side in self.sides:
            if profile in ("widowx", "yam"):
                import xml.etree.ElementTree as ET
                spec = EMBODIMENTS[model].side_arm(side)
                joints = {j.get("name"): j for j in ET.parse(spec.urdf).getroot().findall("joint")}
                names = [f"{side}_{joint}.pos" for joint in spec.joints] + [f"{side}_gripper.pos"]
                limits = [tuple(float(joints[j].find("limit").get(k)) for k in ("lower", "upper"))
                          for j in spec.joints] + [(0., 1.)]
            else:
                names = arm_fields(side)
                limits = REBOT_LIMITS if profile == "rebot" else LIMITS_RAD + [(0., GRIPPER_MAX_MM)]
            self.fields[side] = names
            self.limits.update(zip(names, limits))
        if (set(self.limits) != {f.name for f in cfg.state_schema}
                or set(self.limits) != {f.name for f in cfg.action_schema}):
            raise ValueError(f"hand teleop {profile} requires the matching Portal profile")
        floor = (Region(lo=(None, None, table_z_m)),)
        self._ik = ChunkIK(model, "tcp", hz=fps, obstacles={side: floor for side in self.sides})
        for group in self._ik._groups:  # dream_ik keeps its weights internal (pinned in uv.lock)
            group.w = {**group.w, "ori": jnp.full_like(group.w["ori"], self.TURN_WEIGHT)}
        self._rows = [self._ik.sides.index(side) for side in self.sides]  # the IK orders sides (right, left)
        self.max_joint_rad_s = max_joint_rad_s
        self.rest = self._ik.side_rest[self._rows[0]].copy()

    def clamp_action(self, action):
        if set(action) != set(self.limits) or not all(math.isfinite(v) for v in action.values()):
            raise ValueError("hand action must contain all profile fields with finite values")
        return {name: float(np.clip(value, *self.limits[name])) for name, value in action.items()}

    def warmup(self) -> None:
        """Compile the one-tick solve and the FK (several seconds the first time, cached after)."""
        self._ik.warmup(1)
        for side in self.sides:
            self.fk(self.rest, side)

    def fk(self, q: np.ndarray, side: str) -> np.ndarray:
        return self._ik.fk(np.asarray(q, dtype=np.float64), side)

    def step(self, goals: dict[str, np.ndarray], q: dict[str, np.ndarray]
             ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """One tick toward each side's goal (4, 4) from its joints `q` (6,). Returns the joints to
        command (speed capped, floor fenced) and the tool pose the unconstrained answer reaches."""
        targets, q0 = np.empty((len(self.sides), 1, 4, 4)), np.empty((len(self.sides), 6))
        for side, row in zip(self.sides, self._rows):
            targets[row, 0], q0[row] = goals[side], q[side]
        targets = self._ik.clip_to_region(targets, self._ik.collision_margin)
        result = self._ik.solve(targets, q0, self.max_joint_rad_s)
        command, _ = self._ik.fence_region(result.q, q0, slack=0.01)
        return ({side: command[row, 0].copy() for side, row in zip(self.sides, self._rows)},
                {side: self.fk(result.q_solution[row, 0], side) for side, row in zip(self.sides, self._rows)})
