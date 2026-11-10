"""Imported visual assets, physical receptacles, and deterministic prop resets."""
import hashlib
import json
import unittest

import numpy as np
from livekit.portal import RobotConfig

from utils.common import portal_config
from utils.mujoco_sim import Simulation
from utils.sim_objects import ASSET_DIR


class RoboCasaTests(unittest.TestCase):
    def test_vendored_sources_match_manifest(self):
        manifest = json.loads((ASSET_DIR / 'manifest.json').read_text())
        for name, config in manifest['objects'].items():
            for filename, digest in config['sha256'].items():
                with self.subTest(object=name, file=filename):
                    self.assertEqual(hashlib.sha256((ASSET_DIR / name / filename).read_bytes()).hexdigest(), digest)

    def test_visual_collision_separation_settling_and_reset_on_both_arms(self):
        manifest = json.loads((ASSET_DIR / 'manifest.json').read_text())
        for arms in ('widowx', 'yam'):
            sim = Simulation(portal_config(RobotConfig, 'test', arms), arms=arms, render=False)
            try:
                for name, config in manifest['objects'].items():
                    collision_ids = [sim.model.geom(f'{name}/collision_{i}').id
                                     for i in range(config['collision_pieces'])]
                    self.assertGreaterEqual(len(collision_ids), 1)  # preserve original contact pieces
                    self.assertTrue((sim.model.geom_group[collision_ids] == 3).all())
                    self.assertTrue((sim.model.geom_contype[collision_ids] != 0).all())
                    visual = sim.model.geom(f'{name}/visual_0')
                    self.assertEqual(int(visual.group[0]), 2)
                    self.assertEqual(int(visual.contype[0]), 0)
                    self.assertEqual(int(visual.conaffinity[0]), 0)
                    self.assertGreaterEqual(int(visual.matid[0]), 0)
                    if config['mass_kg']:
                        self.assertAlmostEqual(sim.model.body(f'{name}/object').mass[0], config['mass_kg'])
                for setup in ('workbench',):
                    sim.reset(setup)
                    initial = {name:sim.data.body(f'{name}/object').xpos.copy()
                               for name in manifest['objects']}
                    for _ in range(300):
                        sim.tick()
                    self.assertFalse(sim.data.warning.number.any())
                    self.assertAlmostEqual(sim.data.time, 10.)
                    for name, position in initial.items():
                        np.testing.assert_allclose(sim.data.body(f'{name}/object').xpos, position, atol=.02)
                    # Rounded fruit and curved utensils may roll/tilt before settling.
                    settled = {name: sim.data.body(f'{name}/object').xpos.copy() for name in initial}
                    for _ in range(30):
                        sim.tick()
                    for name, position in settled.items():
                        np.testing.assert_allclose(sim.data.body(f'{name}/object').xpos, position, atol=.0005)
                    sim.data.joint('mug/free').qpos[0] += .1
                    sim.reset()
                    for name, position in initial.items():
                        np.testing.assert_allclose(sim.data.body(f'{name}/object').xpos, position)
                    self.assertEqual(len(sim.state()), 14)  # no object ground truth in Portal state
            finally:
                sim.close()

    def test_receptacles_have_open_collision_cavities(self):
        import mujoco
        from utils.sim_objects import add_objects

        # The diagnostic cube exists only in this test, never on the workbench.
        for name, xy, ceiling in (('mug', [.18, .298], .04),
                                  ('tray', [.31, -.30], .03),
                                  ('bowl', [.34, .02], .03)):
            with self.subTest(receptacle=name):
                spec = mujoco.MjSpec.from_string(f'''<mujoco>
                  <option timestep=".004166666667" integrator="implicit" />
                  <worldbody>
                    <geom type="plane" size="1 1 .1" />
                    <body name="probe" pos="{xy[0]} {xy[1]} .18">
                      <freejoint /><geom type="box" size=".008 .008 .008" mass=".01" />
                    </body>
                  </worldbody>
                </mujoco>''')
                add_objects(spec)
                model = spec.compile()
                data = mujoco.MjData(model)
                for _ in range(1200):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                position = data.body('probe').xpos
                np.testing.assert_allclose(position[:2], xy, atol=.02)
                self.assertTrue(.005 < position[2] < ceiling, position)
                self.assertFalse(data.warning.number.any())


if __name__ == '__main__':
    unittest.main()
