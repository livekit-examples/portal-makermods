"""YAM composition, gripper units, and model-derived Dream IK; no hardware or API calls."""
import json
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import yourdfpy
from dream import arm_file
from dream.client_ik import check_config
from dream_ik.specs import EMBODIMENTS
from livekit.portal import RobotConfig, OperatorConfig

from policies.dream_1_tasks.yam import ARM_FILE, YamMapping
from utils.common import portal_config
from utils.mujoco_sim import Simulation


class YamTests(unittest.TestCase):
    def setUp(self):
        self.cfg = portal_config(OperatorConfig, "yam-test", "yam")
        self.mapping = YamMapping(self.cfg)
        self.sim = Simulation(portal_config(RobotConfig, "yam-test", "yam"), arms="yam", render=False)

    def tearDown(self):
        self.sim.close()

    def test_motion_grip_units_and_independent_arms(self):
        sim = self.sim
        self.assertEqual(sim.model.nu, 14)
        self.assertEqual(sim.model.neq, 2)
        initial = sim.state()
        self.assertAlmostEqual(initial['left_gripper.pos'], .03 / .0475)
        sim.tick(initial | {'left_joint_1.pos': .35, 'left_gripper.pos': 1., 'right_gripper.pos': 0.})
        for _ in range(90):
            sim.tick()
        state = sim.state()
        self.assertAlmostEqual(state['left_joint_1.pos'], .35, delta=.02)
        self.assertAlmostEqual(state['right_joint_1.pos'], 0., delta=.01)
        self.assertAlmostEqual(state['left_gripper.pos'], 1., delta=.01)
        self.assertAlmostEqual(state['right_gripper.pos'], 0., delta=.01)
        for jaw in ('left', 'right'):
            self.assertAlmostEqual(sim.data.joint(f'left/{jaw}_finger').qpos[0], .0475, delta=.0005)
            self.assertAlmostEqual(sim.data.joint(f'right/{jaw}_finger').qpos[0], 0., delta=.0005)
        sim.reset('workbench')
        for _ in range(300):
            _, _, status = sim.tick()
        self.assertEqual(status['setup'], 'workbench')
        self.assertAlmostEqual(status['sim_time'], 10.)
        self.assertFalse(sim.data.warning.number.any())

    def test_model_order_mapping_and_wrong_profiles(self):
        cfg = SimpleNamespace(state_schema=list(reversed(self.cfg.state_schema)),
                              action_schema=list(reversed(self.cfg.action_schema)), video_tracks=self.cfg.video_tracks)
        mapping = YamMapping(cfg)
        state = self.sim.state()
        row = mapping.to_model(state)
        np.testing.assert_allclose(row[:7], [0, 1.1, 1.2, -.65, 0, 0, .03 / .0475])
        self.assertEqual(mapping.to_robot(row), state)
        self.assertNotIn('scene_camera', mapping.cameras.values())
        for profile in ('widowx', 'rebot', 'metal'):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                YamMapping(portal_config(OperatorConfig, 'test', profile))
        with self.assertRaises(ValueError):
            self.sim.validate_action({f.name:0. for f in portal_config(OperatorConfig, 'test', 'widowx').action_schema})
        with self.assertRaises(ValueError):
            self.mapping.to_robot(np.full(14, np.nan))

    def test_dream_fk_tcp_and_jaw_match_actual_yam(self):
        config = json.loads(ARM_FILE.read_text())
        check_config(self.mapping.policy_config('dreamlabs/dream-1-beta', 20)['ik'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            arm = arm_file.register(str(ARM_FILE))
        spec = EMBODIMENTS[arm].side_arm('left')
        urdf = yourdfpy.URDF.load(str(spec.urdf), load_meshes=False)
        sim = self.sim
        for q in (config['rest'], [.2, .6, .8, .3, -.5, -.2], [-.4, .2, .3, -.4, .1, .6]):
            urdf.update_cfg(dict(zip(spec.joints, np.asarray(q, dtype=float))))
            transform = urdf.get_transform(config['flange'], config['base'])
            for side in ('left', 'right'):
                for joint, value in zip(spec.joints, q):
                    sim.data.joint(f'{side}/{joint}').qpos[0] = value
                sim.mj.mj_forward(sim.model, sim.data)
                flange = sim.data.body(f'{side}/link_6')
                base = sim.data.body(f'{side}/arm')
                rotation = flange.xmat.reshape(3, 3)
                np.testing.assert_allclose(flange.xpos - base.xpos, transform[:3, 3], atol=1e-7)
                np.testing.assert_allclose(rotation, transform[:3, :3], atol=1e-7)
                tcp = flange.xpos + np.einsum('ij,j->i', rotation, config['tcp'])
                np.testing.assert_allclose(tcp, sim.data.site(f'{side}/grasp_site').xpos, atol=1e-7)
                # The IK jaw axis must match the actual direction of finger separation.
                jaw = sim.data.geom(f'{side}/left_pad').xpos - sim.data.geom(f'{side}/right_pad').xpos
                jaw /= np.linalg.norm(jaw)
                expected = np.einsum('ij,j->i', rotation, config['jaw'])
                self.assertAlmostEqual(abs(float(np.dot(jaw, expected))), 1., places=5)


if __name__ == '__main__':
    unittest.main()
