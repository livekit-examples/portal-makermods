"""Real WidowX dynamics + control barriers; no display or LiveKit server required."""
import math
import time
import unittest
from types import SimpleNamespace

import numpy as np
from livekit.portal import RobotConfig

from utils.common import portal_config
from utils.mujoco_sim import Simulation
from robot_sim import ControlInbox


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation(portal_config(RobotConfig, "test", "widowx"), render=False)

    def tearDown(self):
        self.sim.close()

    def test_composed_model_and_profile(self):
        sim = self.sim
        self.assertEqual(sim.model.nu, 14)
        self.assertEqual(len(sim.state()), 14)
        self.assertEqual(sim.model.neq, 2)  # independent finger couplings
        self.assertNotEqual(sim.model.joint("left/joint_0").id, sim.model.joint("right/joint_0").id)
        for name in sim.cameras:
            if name != "scene_camera":
                sim.model.camera(name.replace("_wrist_camera", "/wrist_camera"))

    def test_targets_ramp_and_arms_move_independently(self):
        sim = self.sim
        initial = sim.state()
        target = initial | {"left_joint_0.pos": .6, "left_gripper.pos": 1.}
        sim.tick(target)
        self.assertLessEqual(sim.commanded["left_joint_0.pos"], .050001)
        for _ in range(50):
            state, _, _ = sim.tick()
        self.assertGreater(state["left_joint_0.pos"], .5)
        self.assertAlmostEqual(state["right_joint_0.pos"], initial["right_joint_0.pos"], delta=.01)
        self.assertGreater(state["left_gripper.pos"], .98)
        self.assertTrue(np.isfinite(sim.data.qvel).all())

    def test_wrist_clip_plane_keeps_finger_meshes_whole(self):
        sim = self.sim
        near = sim.model.vis.map.znear * sim.model.stat.extent
        for opening in (0., .022, .044):
            for side in ("left", "right"):
                for jaw in ("left", "right"):
                    sim.data.joint(f"{side}/{jaw}_carriage_joint").qpos[0] = opening
            sim.mj.mj_forward(sim.model, sim.data)
            for side in ("left", "right"):
                camera = sim.data.camera(f"{side}/wrist_camera")
                for jaw in ("left", "right"):
                    mesh = sim.model.mesh(f"{side}/gripper_{jaw}")
                    geom_id = int(np.flatnonzero(sim.model.geom_dataid == mesh.id)[0])
                    geom = sim.data.geom(geom_id)
                    start, count = int(mesh.vertadr[0]), int(mesh.vertnum[0])
                    vertices = sim.model.mesh_vert[start:start + count]
                    world = np.einsum("ij,kj->ik", vertices, geom.xmat.reshape(3, 3)) + geom.xpos
                    depth = -np.einsum("ij,j->i", world - camera.xpos,
                                       camera.xmat.reshape(3, 3)[:, 2])
                    with self.subTest(side=side, jaw=jaw, opening=opening):
                        self.assertGreater(depth.min(), near,
                                           "near clipping plane cuts through the finger mesh")

    def test_invalid_vectors_and_limits(self):
        sim = self.sim
        for invalid in ({}, sim.state() | {"left_joint_0.pos": math.nan}, sim.state() | {"left_joint_0.pos": True}):
            with self.assertRaises(ValueError):
                sim.validate_action(invalid)
        values = sim.validate_action(sim.state() | {"left_joint_0.pos": 100., "left_gripper.pos": -5.})
        self.assertEqual(values["left_joint_0.pos"], sim.limits["left_joint_0.pos"][1])
        self.assertEqual(values["left_gripper.pos"], 0.)

    def test_reset_restores_objects_and_discards_target(self):
        sim = self.sim
        initial_apple = sim.data.body("apple/object").xpos.copy()
        sim.tick(sim.state() | {"left_joint_0.pos": 1.})
        for _ in range(20):
            sim.tick()
        sim.data.joint("apple/free").qpos[0] += .1
        sim.reset()
        np.testing.assert_allclose(sim.data.body("apple/object").xpos, initial_apple)
        self.assertEqual(sim.target["left_joint_0.pos"], 0.)
        self.assertEqual(sim.episode, 2)
        self.assertEqual(sim.data.time, 0.)

    def test_hold_cancels_unreached_target(self):
        sim = self.sim
        sim.tick(sim.state() | {"left_joint_0.pos": 1.})
        sim.hold()
        held = sim.state()["left_joint_0.pos"]
        for _ in range(30):
            state, _, _ = sim.tick()
        self.assertAlmostEqual(state["left_joint_0.pos"], held, delta=.02)
        self.assertLess(sim.target["left_joint_0.pos"], .1)


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000
        self.active = "human"
        robot = SimpleNamespace(now_us=lambda: self.now, active_operator=lambda: self.active)
        self.box = ControlInbox(robot, lambda values: values.copy())
        self.box.invalidate()
        self.now += 10_000

    def action(self, **kwargs):
        return SimpleNamespace(**(dict(sender="human", timestamp_us=self.now,
                                       in_reply_to_ts_us=self.now - 1, values={"joint": .1}) | kwargs))

    def test_handoff_rejects_old_sender_and_anchor(self):
        self.box.accept(self.action())
        self.assertIsNotNone(self.box.action)
        self.active = "policy"
        self.box.invalidate()
        self.now += 10_000
        self.box.accept(self.action())
        self.assertIsNone(self.box.action)
        self.box.accept(self.action(sender="policy", in_reply_to_ts_us=1_000_001))
        self.assertIsNone(self.box.action)
        self.box.accept(self.action(sender="policy"))
        self.assertIsNotNone(self.box.action)

    def test_reset_blocks_actions_and_timeout_releases_once(self):
        self.box.busy = True
        self.box.accept(self.action())
        self.assertIsNone(self.box.action)
        self.box.busy = False
        self.box.accept(self.action())
        self.box.received = time.monotonic() - 1
        self.assertTrue(self.box.expired())
        self.box.invalidate()
        self.assertFalse(self.box.expired())
        self.assertTrue(self.box.take()["hold"])
        self.assertFalse(self.box.take()["hold"])

    def test_unanchored_future_and_stale_actions_rejected(self):
        for anchor in (None, self.now + 1, self.now - 800_000):
            self.box.accept(self.action(in_reply_to_ts_us=anchor))
            self.assertIsNone(self.box.action)


if __name__ == "__main__":
    unittest.main()
