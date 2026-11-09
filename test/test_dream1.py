"""Dream-1 observation and action mapping; no robot, network, or API key.

    uv run python -m unittest test.test_dream1
"""
import base64
import importlib.util
import unittest

import cv2
import dream
import numpy as np
from livekit.portal import OperatorConfig

from utils.common import portal_config, ROOT
from utils.metal import GRIPPER_MAX_MM, LIMITS_RAD, arm_fields

_spec = importlib.util.spec_from_file_location("dream1", ROOT / "src/policies/dream_1/run.py")
dream1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dream1)


class Dream1Tests(unittest.TestCase):
    def test_cameras_and_fields_match_portal_yaml(self):
        cfg = portal_config(OperatorConfig, "test", "metal")
        self.assertEqual(set(dream1.CAMERAS.values()), set(cfg.video_tracks))
        self.assertEqual({f.name for f in cfg.action_schema}, set(arm_fields("left") + arm_fields("right")))

    def test_embodiment_spans_are_contiguous(self):
        state = dream1.embodiment()["state"]
        self.assertEqual(state["right_arm"], [0, 6])
        self.assertEqual(state["left_eef_rot6"][1], 32)

    def test_frame_is_cropped_resized_and_bgr(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[:60] = 255  # outside the 16:9 crop
        rgb[60:420, :, 0] = 255  # red inside it
        bgr = cv2.imdecode(np.frombuffer(base64.b64decode(dream1.encode_frame(rgb)), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(bgr.shape, (176, 320, 3))
        self.assertGreater(bgr[88, 160, 2], 200)  # red stays red
        self.assertLess(bgr[2, 160, 1], 50)  # the white band was cropped away

    def test_units_round_trip(self):
        left, right = [0.1, -1.2, 1.5, 0.2, -0.3, 0.4, 40.0], [-0.1, -0.8, 1.0, -0.2, 0.3, -0.4, 80.0]
        state = dict(zip(arm_fields("left") + arm_fields("right"), left + right))
        row = dream1.to_model(state)
        self.assertAlmostEqual(row[6], 0.4)  # 40 mm on Dream's 100 mm grip scale
        for key, value in dream1.to_robot(row).items():
            self.assertAlmostEqual(value, state[key])

    def test_to_robot_stays_inside_the_limits(self):
        action = dream1.to_robot(np.array([9.0] * 6 + [2.0] + [-9.0] * 6 + [-1.0]))
        for side in ("left", "right"):
            *joints, gripper = arm_fields(side)
            for name, (lo, hi) in zip(joints, LIMITS_RAD):
                self.assertTrue(lo <= action[name] <= hi, name)
            self.assertTrue(0.0 <= action[gripper] <= GRIPPER_MAX_MM)

    def test_chunk_rows_are_left_then_right(self):
        rows = dream1.chunk_rows({"q_l_chunk": np.zeros((48, 7)), "q_r_chunk": np.ones((48, 7))})
        self.assertEqual(rows.shape, (48, 14))
        self.assertTrue((rows[:, :7] == 0).all() and (rows[:, 7:] == 1).all())
        for bad in (np.zeros((48, 6)), np.full((48, 7), np.nan), np.zeros((47, 7))):
            with self.assertRaises(dream.DreamError):
                dream1.chunk_rows({"q_l_chunk": np.zeros((48, 7)), "q_r_chunk": bad})

    def test_network_errors_are_retried_and_bad_replies_are_not(self):
        self.assertEqual(dream1.retry_delay(ConnectionResetError("reset by peer")), 1.0)
        with self.assertRaises(dream.DreamError):
            dream1.retry_delay(dream.DreamActionError("bad chunk"))


if __name__ == "__main__":
    unittest.main()
