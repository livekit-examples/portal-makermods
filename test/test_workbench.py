"""Contact stability, repeatable prop resets, and workbench camera framing."""
import unittest

import numpy as np
from livekit.portal import RobotConfig

from utils.common import portal_config
from utils.mujoco_sim import Simulation


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation(portal_config(RobotConfig, "workbench-test", "widowx"), render=False)

    def tearDown(self):
        self.sim.close()

    def test_desk_items_settle_and_reset_on_both_arms(self):
        names = ("usb_drive", "screwdriver", "notebook")
        for arms in ("widowx", "yam"):
            sim = Simulation(portal_config(RobotConfig, "household-test", arms), arms=arms, render=False)
            try:
                initial = {name: sim.data.body(name).xpos.copy() for name in names}
                for _ in range(300):
                    _, _, status = sim.tick()
                self.assertAlmostEqual(status["sim_time"], 10.)
                self.assertEqual(status["setup"], "workbench")
                self.assertFalse(sim.data.warning.number.any())
                for name in names:
                    position = sim.data.body(name).xpos
                    self.assertTrue(.005 < position[2] < .06, (name, position))
                    np.testing.assert_allclose(position[:2], initial[name][:2], atol=.005)
                self.assertLess(abs(sim.data.qvel).max(), .1)
                sim.data.joint("usb_drive/free").qpos[0] += .1
                sim.tick(reset=True, setup="workbench")
                for name in names:
                    np.testing.assert_allclose(sim.data.body(name).xpos, initial[name], atol=.003)
                with self.assertRaises(ValueError):
                    sim.reset("unknown")
                self.assertEqual(sim.setup, "workbench")
            finally:
                sim.close()

    def test_overhead_frames_both_grippers_and_household_items(self):
        sim = self.sim
        camera = sim.data.camera("overhead_camera")
        rotation = camera.xmat.reshape(3, 3)
        tangent = np.tan(np.deg2rad(sim.model.camera("overhead_camera").fovy[0] / 2))
        points = [sim.data.site(f"{side}/ee_site").xpos for side in ("left", "right")]
        points += [sim.data.body(name + "/object").xpos for name in
                   ("apple", "banana", "lemon", "sponge", "soap", "drink_can", "spoon", "bowl")]
        for point in points:
            x, y, z = rotation.T @ (point - camera.xpos)
            self.assertLess(z, 0)
            self.assertLess(abs(y / z), tangent)
            self.assertLess(abs(x / z), tangent * 4 / 3)


if __name__ == "__main__":
    unittest.main()
