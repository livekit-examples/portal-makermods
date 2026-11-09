"""Webcam hand teleop: gesture, hand pose, mapping, IK and the pilot; no camera, model, or robot.

    uv run python -m unittest test.test_hands

Hands are synthetic: a hand of MediaPipe's 21 landmarks, built in the camera frame and projected
through a pinhole, as the model would report it.
"""
import math
import re
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from hand_teleop import Pilot, Sighting, build_pilot
from utils.hands import (CAMERA_TO_BASE, CURLED_BELOW, INDEX_TIP, OPEN_ABOVE, THUMB_TIP, ArmFollower, ArmsIK, Clutch,
                         Hand, HandPose, HandSize, Mapping, OneEuro, assign_sides, camera_to_base, curl, hand_pose,
                         palm_position, palm_rotation, palm_size, pinch, turn_between, view_rotation)
from utils.metal import GRIPPER_MAX_MM, SIDES, arm_fields

FOCAL, CENTER = 457.0, (320.0, 240.0)  # 640x480 at a 70 degree field of view


def hand_world(curled: bool = True, pinch_m: float = 0.08) -> np.ndarray:
    """A right hand, palm to the lens, fingers up (camera -y), about its palm's centre (m): curling the
    fingers or pinching leaves the palm where it is, as it does a real hand's."""
    w = np.zeros((21, 3))
    w[0] = (0.0, 0.04, 0.0)  # wrist
    for base, x in ((5, -0.025), (9, 0.0), (13, 0.02), (17, 0.038)):  # index, middle, ring, pinky
        w[base] = (x, -0.05, 0.0)
        straight = base == 5 or not curled
        for k in (1, 2, 3):
            w[base + k] = (x, -0.05 - 0.03 * k, 0.0) if straight else (x, -0.05 + 0.012 * k, -0.025)
    for k in (1, 2, 3):  # thumb, toward the index tip
        w[k] = (-0.03 - 0.01 * k, 0.03 - 0.02 * k, 0.0)
    w[THUMB_TIP] = w[INDEX_TIP] + (-pinch_m, 0.0, 0.0)
    return w - w[[0, 5, 9, 13, 17]].mean(axis=0)


def looking_along(ray: np.ndarray) -> np.ndarray:
    """The lens turned, by the shortest turn, to look along `ray` (Rodrigues)."""
    z = ray / np.linalg.norm(ray)
    axis = np.cross([0.0, 0.0, 1.0], z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * skew + (1 - z[2]) * skew @ skew


def project(world: np.ndarray, at: np.ndarray, rotation: np.ndarray = np.eye(3)) -> Hand:
    """The hand placed with its centre at `at` (camera frame) and seen through the pinhole. Its world
    landmarks are as MediaPipe reads them off a crop around the hand: as seen looking along its ray."""
    points = world @ rotation.T + at
    image = points[:, :2] / points[:, 2:] * FOCAL + CENTER
    crop = points[[0, 5, 9, 13, 17]].mean(axis=0)  # MediaPipe centres its crop on the palm
    return Hand("right", image, world @ rotation.T @ looking_along(crop))


def rot(axis: str, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


class GestureTests(unittest.TestCase):
    def test_curl_tells_a_gripper_shape_from_an_open_hand(self):
        self.assertTrue((curl(hand_world(curled=True)) < CURLED_BELOW).all())
        self.assertTrue((curl(hand_world(curled=False)) > OPEN_ABOVE).all())

    def test_pinch_is_the_thumb_to_index_gap(self):
        self.assertAlmostEqual(pinch(hand_world(pinch_m=0.05)), 0.05)

    def test_clutch_closes_after_three_curled_frames_and_opens_after_two_open(self):
        clutch, curled, open_ = Clutch(), np.array([0.8, 0.7, 0.9]), np.array([0.8, 1.8, 0.9])
        self.assertEqual([clutch.update(curled) for _ in range(3)], [False, False, True])
        self.assertTrue(clutch.update(np.array([0.8, 1.1, 0.9])), "between the thresholds keeps it closed")
        self.assertEqual([clutch.update(open_) for _ in range(2)], [True, False])
        clutch.update(curled), clutch.update(curled)
        self.assertFalse(clutch.update(np.array([0.8, 1.05, 0.9])), "closing needs every finger in the palm")
        for _ in range(3):
            clutch.update(curled)
        self.assertFalse(clutch.update(None), "a lost hand opens it at once")

    def test_clutch_needs_the_fingers_all_the_way_into_the_palm(self):
        clutch = Clutch()
        hooked = np.array([1.2, 1.2, 1.25])  # bent, tips short of the palm
        self.assertFalse(any(clutch.update(hooked) for _ in range(10)), "a half curl never takes the arm")
        for _ in range(3):
            clutch.update(np.array([0.8, 0.7, 0.9]))
        self.assertEqual([clutch.update(hooked) for _ in range(2)], [True, False], "coming out of the palm lets go")

    def test_sides_follow_the_label_or_position(self):
        a = Hand("right", np.full((21, 2), 100.0), np.zeros((21, 3)))  # on the raw image's left
        b = Hand("left", np.full((21, 2), 500.0), np.zeros((21, 3)))
        self.assertEqual(assign_sides([b, a]), {"right": a, "left": b})
        self.assertEqual(assign_sides([a])["right"], a)
        self.assertEqual(assign_sides([a, b], swap=True)["left"].image[0, 0], 100.0)
        same = Hand("right", b.image, b.world)
        self.assertEqual(assign_sides([same, a]), {"right": a, "left": same},
                         "one label for two hands: the one on the raw image's left is the operator's right")


class PoseTests(unittest.TestCase):
    def test_palm_position_finds_the_hand_and_how_far_it_moved(self):
        world = hand_world()
        palm = world[[0, 5, 9, 13, 17]].mean(axis=0)
        found = {}
        for at in ([0.0, 0.0, 0.5], [0.12, -0.05, 0.4], [-0.1, 0.08, 0.7]):
            hand = project(world, np.array(at))
            found[tuple(at)] = palm_position(hand.image, hand.world, FOCAL, CENTER)
            np.testing.assert_allclose(found[tuple(at)], palm + at, atol=0.02 * at[2])
        # What the arm follows is the motion: a few percent of it at most.
        a, b = np.array([0.0, 0.0, 0.5]), np.array([-0.1, 0.08, 0.7])
        moved = found[tuple(b)] - found[tuple(a)]
        self.assertLess(np.linalg.norm(moved - (b - a)), 0.03 * np.linalg.norm(b - a))

    def test_depth_holds_through_a_wandering_scale_and_a_pinch(self):
        at = np.array([0.05, -0.03, 0.55])
        hand = project(hand_world(), at, rot("y", 20))
        size = palm_size(hand.world)
        found = palm_position(hand.image, hand.world, FOCAL, CENTER, size)
        np.testing.assert_allclose(found, at, atol=1e-3)
        for scale in (0.95, 1.05):  # MediaPipe's metric scale, a few percent off in one frame
            np.testing.assert_allclose(palm_position(hand.image, hand.world * scale, FOCAL, CENTER, size), found,
                                       atol=1e-9, err_msg="a steady palm size keeps the depth")
            own = palm_position(hand.image, hand.world * scale, FOCAL, CENTER)
            self.assertGreater(abs(own[2] - found[2]), 0.02, "the frame's own scale puts it centimetres off")
        pinched = project(hand_world(pinch_m=0.02), at, rot("y", 20))
        np.testing.assert_allclose(palm_position(pinched.image, pinched.world, FOCAL, CENTER, size), found,
                                   atol=1e-9, err_msg="the fingers do not count: a pinch never moves the arm")

    def test_hand_size_is_the_median_of_recent_sightings(self):
        sizes, world = HandSize(keep=5), hand_world()
        self.assertIsNone(sizes.value)
        for scale in (3.0, 1.0, 1.1, 0.9, 1.0, 1.02):  # the first falls out; one bad frame never counts
            sizes.add(world * scale)
        self.assertAlmostEqual(sizes.value, palm_size(world) * 1.0)

    def test_palm_rotation_turns_with_the_hand(self):
        world, turn = hand_world(), rot("y", 30) @ rot("x", -20)
        before, after = palm_rotation(world), palm_rotation(world @ turn.T)
        np.testing.assert_allclose(before.T @ before, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(before), 1.0)
        np.testing.assert_allclose(after @ before.T, turn, atol=1e-9)

    def test_view_rotation_points_the_lens_along_the_ray(self):
        np.testing.assert_allclose(view_rotation(CENTER, FOCAL, CENTER), np.eye(3), atol=1e-12)
        pixel = (600.0, 50.0)
        ray = np.array([(pixel[0] - CENTER[0]) / FOCAL, (pixel[1] - CENTER[1]) / FOCAL, 1.0])
        view = view_rotation(pixel, FOCAL, CENTER)
        np.testing.assert_allclose(view @ [0, 0, 1], ray / np.linalg.norm(ray), atol=1e-12)
        np.testing.assert_allclose(view, looking_along(ray), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(view), 1.0)

    def test_a_hand_off_the_image_centre_reads_as_turned_as_it_is(self):
        world, held = hand_world(), rot("y", 20) @ rot("x", -10)
        centre = project(world, np.array([0.0, 0.0, 0.5]), held)
        edge = project(world, np.array([0.2, -0.12, 0.5]), held)  # about 25 degrees off the lens's axis
        self.assertGreater(math.degrees(turn_between(palm_rotation(centre.world), palm_rotation(edge.world))), 20,
                           "MediaPipe's landmarks alone read a hand slid to the edge as turned")
        a, b = (hand_pose(h, 0.0, FOCAL, CENTER).rotation for h in (centre, edge))
        self.assertLess(math.degrees(turn_between(a, b)), 1.0, "turned back along the ray, it is not")
        np.testing.assert_allclose(a, held @ palm_rotation(world), atol=1e-3)

    def test_camera_motion_maps_to_the_operators_point_of_view(self):
        to_base = camera_to_base(0.0)
        np.testing.assert_allclose(to_base @ [0, 0, -1], [1, 0, 0], err_msg="toward the screen reaches forward")
        np.testing.assert_allclose(to_base @ [-1, 0, 0], [0, -1, 0],
                                   err_msg="the operator's right (the raw image's left) is the robot's right")
        np.testing.assert_allclose(to_base @ [0, -1, 0], [0, 0, 1], err_msg="up is up")
        self.assertAlmostEqual(np.linalg.det(CAMERA_TO_BASE), 1.0)
        np.testing.assert_allclose(camera_to_base(90.0) @ [0, 0, -1], [0, -1, 0], atol=1e-12,
                                   err_msg="an arm turned 90 degrees left sees forward as its right")

    def test_one_euro_quiets_jitter_and_follows_motion(self):
        f, rng = OneEuro(1.0, 4.0), np.random.default_rng(0)
        f.reset([0.5], 0.0)
        still = [f([0.5 + rng.normal(0, 0.003)], k / 30)[0] for k in range(1, 91)]
        self.assertLess(np.std(still[30:]), 0.0012, "3 mm of jitter on a still hand cut to about a third")
        moving = [f([0.5 + 0.3 * k / 30], 3 + k / 30)[0] for k in range(1, 31)]  # 30 cm/s for a second
        self.assertGreater(moving[-1], 0.5 + 0.3 - 0.02, "less than 2 cm behind a 30 cm/s hand")


def pose(at: float, position, rotation=np.eye(3), pinch_m=0.08) -> HandPose:
    return HandPose(at, np.asarray(position, dtype=np.float64), rotation, pinch_m, np.ones(3))


EXACT = {"fine_scale": 1.0, "hold_m": 0.0, "hold_rad": 0.0}  # the mapping's geometry alone


class FollowerTests(unittest.TestCase):
    def follow(self, rotate="off", grip_mm=50.0, pinch_m=0.06, **mapping):
        """A follower gripped at a hand 50 cm from the lens; `EXACT` unless `mapping` says otherwise."""
        self.clock = 0.0
        follower = ArmFollower(Mapping(to_base=camera_to_base(), rotate=rotate, **{**EXACT, **mapping}))
        tcp = np.eye(4)
        tcp[:3, 3] = (0.3, 0.0, 0.3)
        follower.observe(pose(0.0, [0.0, 0.0, 0.5], pinch_m=pinch_m))
        follower.engage(tcp, grip_mm)
        return follower

    def run_to(self, follower, position, rotation=np.eye(3), pinch_m=0.06, seconds=3.0):
        for _ in range(int(30 * seconds)):
            self.clock += 1 / 30
            follower.observe(pose(self.clock, position, rotation, pinch_m))
            goal = follower.goal(30)
        return goal

    def test_the_tool_moves_by_the_hands_motion(self):
        follower = self.follow()
        goal = self.run_to(follower, [-0.05, -0.02, 0.45])  # right, up, toward the screen
        np.testing.assert_allclose(goal[:3, 3], [0.35, -0.05, 0.32], atol=1e-3)
        np.testing.assert_allclose(goal[:3, :3], np.eye(3), atol=1e-9, err_msg="rotate off keeps the orientation")

    def test_the_target_moves_no_faster_than_the_limit(self):
        follower = self.follow()
        follower.observe(pose(1 / 30, [0.0, 0.0, 0.2]))  # 30 cm at once
        step = np.linalg.norm(follower.goal(30)[:3, 3] - [0.3, 0.0, 0.3])
        self.assertLessEqual(step, 0.5 / 30 + 1e-9)

    def test_yaw_keeps_only_the_turn_about_vertical(self):
        follower = self.follow(rotate="yaw", turn_deadband_rad=0.0)
        # The hand turns about the camera's vertical (-y) by 30 degrees and tilts 20 about x.
        goal = self.run_to(follower, [0.0, 0.0, 0.5], rot("y", -30) @ rot("x", 20))
        self.assertAlmostEqual(goal[2, 2], 1.0, places=6, msg="no tilt")
        self.assertAlmostEqual(math.degrees(math.atan2(goal[1, 0], goal[0, 0])), 30.0, delta=6.0)

    def test_full_rotation_maps_the_turn_into_the_base(self):
        self.assertEqual(Mapping(to_base=CAMERA_TO_BASE).rotate, "full", "the gripper turns by default")
        follower = self.follow(rotate="full")
        goal = self.run_to(follower, [0.0, 0.0, 0.5], rot("x", 25))
        expected = CAMERA_TO_BASE @ rot("x", 25 - 5) @ CAMERA_TO_BASE.T  # less the 5 degree deadband
        np.testing.assert_allclose(goal[:3, :3], expected, atol=1e-3)
        np.testing.assert_allclose(goal[:3, 3], [0.3, 0.0, 0.3], atol=1e-6, err_msg="about the tool's own point")
        self.assertAlmostEqual(math.degrees(follower.turned()), 20.0, places=1)

    def test_a_hand_that_only_moves_keeps_the_tool_steady(self):
        follower = self.follow(rotate="full")
        goal = self.run_to(follower, [-0.05, 0.0, 0.5], rot("y", 4) @ rot("z", -3))  # a few degrees of drift
        np.testing.assert_allclose(goal[:3, :3], np.eye(3), atol=1e-9)
        self.assertAlmostEqual(goal[1, 3], -0.05, places=3)

    def test_turning_past_the_cap_takes_the_anchor_along(self):
        follower = self.follow(rotate="full")
        self.run_to(follower, [0.0, 0.0, 0.5], rot("z", 60))
        goal = self.run_to(follower, [0.0, 0.0, 0.5], rot("z", 120))
        self.assertAlmostEqual(math.degrees(turn_between(np.eye(3), goal[:3, :3])), 90 - 5, delta=0.5,
                               msg="stops at the cap, less the deadband")
        goal = self.run_to(follower, [0.0, 0.0, 0.5], rot("z", 100))
        self.assertAlmostEqual(math.degrees(turn_between(np.eye(3), goal[:3, :3])), 70 - 5, delta=0.5,
                               msg="turning back acts at once")
        axis = CAMERA_TO_BASE @ [0, 0, 1]  # the lens's axis: the robot's backward
        np.testing.assert_allclose(goal[:3, :3] @ axis, axis, atol=1e-6, err_msg="about the same axis throughout")

    def test_the_anchor_slides_where_the_arm_cannot_follow(self):
        follower = self.follow()
        goal = self.run_to(follower, [0.0, 0.1, 0.5])  # 10 cm down, into a floor at z = 0.25
        floor = goal.copy()
        floor[2, 3] = 0.25
        follower.settle(floor)
        goal = self.run_to(follower, [0.0, 0.08, 0.5])  # 2 cm back up
        self.assertAlmostEqual(goal[2, 3], 0.27, places=3, msg="rises at once, not after 5 cm of dead zone")

    def test_the_gripper_waits_for_the_pinch_to_reach_it(self):
        follower = self.follow(grip_mm=0.0, pinch_m=0.09)  # holding something; the hand comes back open
        self.assertTrue(follower.grip_waiting)
        self.run_to(follower, [0.0, 0.0, 0.5], pinch_m=0.09, seconds=0.5)
        self.assertEqual(follower.grip_mm, 0.0, "an open hand does not drop it")
        self.run_to(follower, [0.0, 0.0, 0.5], pinch_m=0.02, seconds=0.5)
        self.run_to(follower, [0.0, 0.0, 0.5], pinch_m=0.09, seconds=1.0)
        self.assertGreater(follower.grip_mm, 0.9 * GRIPPER_MAX_MM, "once taken, the pinch drives it")


class SteadinessTests(unittest.TestCase):
    """The defaults against a jittery hand: what holds a still arm still and makes small moves easier."""

    def gripped(self, rotate="full", **mapping):
        self.clock, self.rng = 0.0, np.random.default_rng(1)
        follower = ArmFollower(Mapping(to_base=camera_to_base(), rotate=rotate, **mapping))
        for _ in range(30):  # a second in view before the grip: it anchors to the settled hand, not to one frame
            self.see(follower, [0.0, 0.0, 0.5])
        tcp = np.eye(4)
        tcp[:3, 3] = (0.3, 0.0, 0.3)
        follower.engage(tcp, 40.0)
        return follower, tcp

    def see(self, follower, position, rotation=np.eye(3)):
        """One sighting with webcam-like jitter: 1 mm across the image, 6 mm in depth, 3 degrees of turn."""
        self.clock += 1 / 30
        jitter = self.rng.normal(0, 1, 3) * [0.001, 0.001, 0.006]
        wobble = Rotation.from_rotvec(self.rng.normal(0, math.radians(3), 3)).as_matrix()
        follower.observe(pose(self.clock, np.asarray(position) + jitter, rotation @ wobble))
        return follower.goal(30) if follower.engaged else None

    def wander(self, **mapping):
        follower, tcp = self.gripped(**mapping)
        places = np.array([self.see(follower, [0.0, 0.0, 0.5])[:3, 3] for _ in range(150)])
        return float(np.ptp(places, axis=0).max()), float(np.abs(places - tcp[:3, 3]).max())

    def test_a_still_hand_holds_the_arm_still(self):
        steady, off = self.wander()
        jittery, _ = self.wander(**EXACT)
        self.assertLess(steady, 0.003, "under 3 mm over five seconds")
        self.assertLess(steady, jittery / 2, f"half or less of the jitter's own {jittery * 1000:.1f} mm")
        self.assertLess(off, 0.003, "and no jump when the grip closes")

    def wobble(self, **mapping):
        follower, _ = self.gripped(**mapping)
        for _ in range(90):  # turn 30 degrees, past the deadband, and hold
            self.see(follower, [0.0, 0.0, 0.5], rot("x", 30))
        turns = [self.see(follower, [0.0, 0.0, 0.5], rot("x", 30))[:3, :3] for _ in range(150)]
        return max(math.degrees(turn_between(turns[0], t)) for t in turns)

    def test_a_turned_hand_held_still_holds_the_gripper_steady(self):
        steady, jittery = self.wobble(), self.wobble(**EXACT)
        self.assertLess(steady, 2.5, "against 3 degrees of jitter a frame on each axis")
        self.assertLess(steady, 0.7 * jittery, f"the hold takes a third or more off {jittery:.1f} degrees")

    def test_a_slow_hand_moves_the_tool_finely_a_quick_one_fully(self):
        follower = ArmFollower(Mapping(to_base=camera_to_base(), rotate="off", hold_m=0.0))
        tcp = np.eye(4)
        tcp[:3, 3] = (0.3, 0.0, 0.3)
        follower.observe(pose(0.0, [0.0, 0.0, 0.5]))
        follower.engage(tcp, 40.0)
        clock = 0.0
        for k in range(90):  # 2 cm to the operator's right at 1 cm/s
            clock += 1 / 30
            follower.observe(pose(clock, [-0.02 * min(1.0, (k + 1) / 60), 0.0, 0.5]))
            slow = follower.goal(30)[1, 3]
        self.assertAlmostEqual(slow, -0.4 * 0.02, delta=0.001, msg="a careful hand moves it at the fine scale")
        for k in range(60):  # 15 cm more at 30 cm/s
            clock += 1 / 30
            follower.observe(pose(clock, [-0.02 - 0.15 * min(1.0, (k + 1) / 15), 0.0, 0.5]))
            quick = follower.goal(30)[1, 3]
        self.assertAlmostEqual(quick - slow, -0.15, delta=0.005, msg="a quick one at full scale")


class IKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ik = ArmsIK(30, table_z_m=0.0, max_joint_rad_s=1.5)
        cls.ik.warmup()

    def test_each_side_moves_only_its_own_arm(self):
        q = {side: self.ik.rest.copy() for side in SIDES}
        goals = {side: self.ik.fk(q[side], side) for side in SIDES}
        goals["left"][2, 3] += 0.01
        for _ in range(15):
            q, _ = self.ik.step(goals, q)
        self.assertAlmostEqual(self.ik.fk(q["left"], "left")[2, 3], goals["left"][2, 3], delta=0.002)
        np.testing.assert_allclose(q["right"], self.ik.rest, atol=1e-3)

    def test_a_tilt_the_arm_cannot_make_gives_way_not_the_place(self):
        follower = ArmFollower(Mapping(to_base=camera_to_base(), rotate="full"))
        q = {side: self.ik.rest.copy() for side in SIDES}
        start = self.ik.fk(q["right"], "right")
        follower.observe(pose(0.0, [0.0, 0.0, 0.5]))
        follower.engage(start, 40.0)
        clock, places = 0.0, []
        for degrees in (65, 0):  # tip the hand forward past what the arm can do from its rest pose, then back
            for k in range(90):
                clock += 1 / 30
                tilt = degrees if k >= 30 else 65 * (k + 1) / 30 if degrees else 65 * (1 - (k + 1) / 30)
                follower.observe(pose(clock, [0.0, 0.0, 0.5], rot("x", tilt)))
                goals = {side: self.ik.fk(q[side], side) for side in SIDES}
                goals["right"] = follower.goal(30)
                q, reachable = self.ik.step(goals, q)
                follower.settle(reachable["right"])
                places.append(np.linalg.norm(self.ik.fk(q["right"], "right")[:3, 3] - start[:3, 3]))
            if degrees:
                turned = turn_between(start[:3, :3], self.ik.fk(q["right"], "right")[:3, :3])
        end = self.ik.fk(q["right"], "right")
        self.assertGreater(math.degrees(turned), 20, "the gripper tips as far as it can")
        self.assertLess(max(places), 0.01, "the tool stays about where the hand put it (11 cm off at 50:10 weights)")
        self.assertLess(places[-1], 0.001, "and comes back to it")
        self.assertLess(math.degrees(turn_between(start[:3, :3], end[:3, :3])), 2.5,
                        "level again with the hand, give or take the 1.5 degree hold")

    def test_the_floor_holds_the_tool_above_the_table(self):
        q = {side: self.ik.rest.copy() for side in SIDES}
        goals = {side: self.ik.fk(q[side], side) for side in SIDES}
        goals["right"][2, 3] = -0.2
        for _ in range(90):
            q, reachable = self.ik.step(goals, q)
        self.assertGreater(self.ik.fk(q["right"], "right")[2, 3], 0.0)
        self.assertGreater(reachable["right"][2, 3], 0.0)


class RebotPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ik = ArmsIK(30, table_z_m=0.0, max_joint_rad_s=0.5, profile="rebot")
        cls.ik.warmup()

    def setUp(self):
        self.pilot = build_pilot(30, self.ik, "off", 640)
        self.state = dict(zip(arm_fields("right"), [0.0, -0.6, -1.0, 0.7, 0.0, 0.0, 0.4]))
        self.seq = 0

    def sight(self, label="right", pinch_m=0.09):
        self.seq += 1
        now = 100.0 + self.seq / 30
        hand = project(hand_world(pinch_m=pinch_m), np.array([0.0, 0.0, 0.5]))
        hand = Hand(label, hand.image, hand.world)
        self.pilot.see(Sighting(self.seq, now, np.zeros((480, 640, 3), np.uint8), [hand]), now)

    def test_single_arm_ik_and_two_camera_contract(self):
        from livekit.portal import OperatorConfig
        from utils.common import portal_config
        cfg = portal_config(OperatorConfig, "test", "rebot")
        self.assertEqual(set(cfg.video_tracks), {"overhead_camera", "arm_camera"})
        self.assertEqual(self.ik._ik.sides, ("right",))
        for _ in range(3):
            self.sight()
        action = self.pilot.tick(True, self.state)
        self.assertEqual(set(action), {f.name for f in cfg.action_schema})
        self.assertAlmostEqual(action["right_gripper.pos"], 0.4, msg="open pinch must not jump the gripper")
        self.assertLess(action["right_joint3.pos"], 0, "use ReBot's negative elbow limits")
        for name in arm_fields("right")[:6]:
            self.assertLessEqual(abs(action[name] - self.state[name]), 0.5 / 30 + 1e-6)

    def test_left_hand_and_incomplete_state_never_drive(self):
        for _ in range(3):
            self.sight(label="left")
        self.assertIsNone(self.pilot.tick(True, self.state))
        for _ in range(3):
            self.sight()
        self.assertIsNone(self.pilot.tick(False, self.state))
        self.assertIsNone(self.pilot.tick(True, {"right_gripper.pos": 0.4}))
        self.assertIsNotNone(self.pilot.tick(True, self.state))
        self.pilot.drop()
        self.assertIsNone(self.pilot.tick(True, None))

    def test_gripper_takeover_converts_normalized_units_both_ways(self):
        for _ in range(3):
            self.sight(pinch_m=0.054)  # 40% of the 3..9 cm pinch span
        action = self.pilot.tick(True, self.state)
        self.assertAlmostEqual(action["right_gripper.pos"], 0.4, places=5)
        for _ in range(60):
            self.sight(pinch_m=0.09)
            action = self.pilot.tick(True, self.state)
        self.assertGreater(action["right_gripper.pos"], 0.95)
        self.assertLessEqual(action["right_gripper.pos"], 1.0)


class PilotTests(unittest.TestCase):
    """The whole chain but MediaPipe: synthetic sightings in, Portal actions out."""

    @classmethod
    def setUpClass(cls):
        cls.ik = ArmsIK(30, table_z_m=0.0, max_joint_rad_s=1.5)
        cls.ik.warmup()

    def setUp(self):
        self.pilot = build_pilot(30, self.ik, "off", 640)
        self.state = {}
        for side in SIDES:
            self.state.update(zip(arm_fields(side), [*self.ik.rest, 40.0]))
        self.seq, self.now = 0, 100.0

    def sight(self, *hands: Hand) -> None:
        self.seq += 1
        self.now += 1 / 30
        self.pilot.see(Sighting(self.seq, self.now, np.zeros((480, 640, 3), np.uint8), list(hands)), self.now)

    def tcp(self, action, side):
        return self.ik.fk([action[name] for name in arm_fields(side)[:6]], side)[:3, 3]

    def test_a_gripping_hand_drives_its_arm_and_an_open_one_lets_go(self):
        open_hand = project(hand_world(curled=False), np.array([0.0, 0.0, 0.5]))
        self.sight(open_hand)
        self.assertIsNone(self.pilot.tick(True, self.state), "an open hand sends nothing")
        for _ in range(3):
            self.sight(project(hand_world(), np.array([0.0, 0.0, 0.5])))
        first = self.pilot.tick(True, self.state)
        self.assertIsNotNone(first)
        start = self.tcp(first, "right")
        for _ in range(45):  # 5 cm to the operator's right: the raw image's left, camera -x
            self.sight(project(hand_world(), np.array([-0.05, 0.0, 0.5])))
            action = self.pilot.tick(True, self.state)
        moved = self.tcp(action, "right") - start
        self.assertLess(moved[1], -0.035, f"the right arm moved to the robot's right: {moved}")
        self.assertLess(abs(moved[0]) + abs(moved[2]), 0.015, f"and only that way: {moved}")
        for name in arm_fields("left"):
            self.assertAlmostEqual(action[name], self.state[name], places=5, msg="the left arm holds")
        for _ in range(2):
            self.sight(project(hand_world(curled=False), np.array([-0.05, 0.0, 0.5])))
        self.assertIsNone(self.pilot.tick(True, self.state), "opening the hand stops sending")

    def test_nothing_is_sent_without_control_and_control_restarts_from_the_state(self):
        for _ in range(3):
            self.sight(project(hand_world(), np.array([0.0, 0.0, 0.5])))
        self.assertIsNone(self.pilot.tick(False, self.state))
        self.assertIsNone(self.pilot.tick(True, None), "no state yet: nothing to start from")
        moved = dict(self.state, **{"right_joint1.pos": 0.3})
        self.pilot.drop()
        action = self.pilot.tick(True, moved)
        self.assertAlmostEqual(action["right_joint1.pos"], 0.3, places=3, msg="starts where the arm is now")

    def test_a_jump_lets_go(self):
        for _ in range(3):
            self.sight(project(hand_world(), np.array([0.0, 0.0, 0.5])))
        self.assertIsNotNone(self.pilot.tick(True, self.state))
        self.sight(project(hand_world(), np.array([0.3, 0.0, 0.5])))  # 30 cm in one frame: another hand
        self.assertIsNone(self.pilot.tick(True, self.state))
        self.assertIn("jumped", " ".join(self.pilot.drain_events()))

    def test_a_turning_hand_turns_its_gripper(self):
        self.pilot = build_pilot(30, self.ik, "full", 640)
        at = np.array([0.0, 0.0, 0.5])
        for _ in range(3):
            self.sight(project(hand_world(), at))
        first = self.pilot.tick(True, self.state)
        start = self.ik.fk([first[name] for name in arm_fields("right")[:6]], "right")
        for k in range(60):  # roll 30 degrees about the lens's axis in a second, then hold
            self.sight(project(hand_world(), at, rot("z", 30 * min(1.0, (k + 1) / 30))))
            action = self.pilot.tick(True, self.state)
        now = self.ik.fk([action[name] for name in arm_fields("right")[:6]], "right")
        turn = now[:3, :3] @ start[:3, :3].T
        expected = CAMERA_TO_BASE @ rot("z", 30 - 5) @ CAMERA_TO_BASE.T
        self.assertLess(math.degrees(turn_between(expected, turn)), 3.0,
                        f"rolled 25 degrees about the robot's forward axis: {np.round(turn, 3)}")
        self.assertLess(np.linalg.norm(now[:3, 3] - start[:3, 3]), 0.02, "and stayed about where it was")
        shown = float(re.search(r"turn (\d+) deg", " ".join(self.label())).group(1))
        self.assertAlmostEqual(shown, 25 - 1.5, delta=1.5, msg="the label shows the turn (less the 1.5 degree hold)")

    def test_a_wandering_scale_does_not_move_the_arm(self):
        at = np.array([0.0, 0.0, 0.5])
        for _ in range(30):  # in view for a second, then grip
            self.sight(project(hand_world(), at))
        first = self.pilot.tick(True, self.state)
        start = self.tcp(first, "right")
        for k in range(90):  # MediaPipe's metric scale wanders 4% either way: 2 cm of depth, from size alone
            hand = project(hand_world(), at)
            self.sight(Hand(hand.label, hand.image, hand.world * (1 + 0.04 * math.sin(k / 4))))
            action = self.pilot.tick(True, self.state)
        self.assertLess(np.linalg.norm(self.tcp(action, "right") - start), 0.002)

    def test_a_flip_lets_go_when_the_gripper_turns(self):
        held = self.pilot  # rotate "off": a flip changes nothing it follows
        turning = build_pilot(30, self.ik, "full", 640)
        for pilot in (held, turning):
            self.pilot = pilot
            for _ in range(3):
                self.sight(project(hand_world(), np.array([0.0, 0.0, 0.5])))
            self.assertIsNotNone(pilot.tick(True, self.state))
            self.sight(project(hand_world(), np.array([0.0, 0.0, 0.5]), rot("z", 90)))  # a quarter turn in a frame
        self.assertIsNotNone(held.tick(True, self.state), "rotate off keeps following")
        self.assertIsNone(turning.tick(True, self.state))
        self.assertIn("flipped", " ".join(turning.drain_events()))

    def label(self) -> list[str]:
        """The text `draw` would put above each hand, read back from the drawing calls."""
        import hand_teleop
        texts, original = [], hand_teleop.caption
        hand_teleop.caption = lambda image, text, *args, **kwargs: texts.append(text)
        try:
            hand_teleop.draw(Sighting(self.seq, self.now, np.zeros((480, 640, 3), np.uint8), []), self.pilot, "")
        finally:
            hand_teleop.caption = original
        return texts

    def test_a_stale_webcam_lets_go(self):
        for _ in range(3):
            self.sight(project(hand_world(), np.array([0.0, 0.0, 0.5])))
        self.assertIsNotNone(self.pilot.tick(True, self.state))
        self.pilot.see(Sighting(self.seq, self.now, np.zeros((480, 640, 3), np.uint8), []), self.now + 1.0)
        self.assertIsNone(self.pilot.tick(True, self.state))


if __name__ == "__main__":
    unittest.main()
