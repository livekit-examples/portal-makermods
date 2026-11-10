"""WidowX units and actual MuJoCo/URDF kinematics; no Dream API requests."""
import json
import unittest
import warnings
from types import SimpleNamespace

import dream
import mujoco
import numpy as np
import yourdfpy
from dream import arm_file
from dream.client_ik import check_config
from dream_ik.specs import EMBODIMENTS
from livekit.portal import OperatorConfig

from policies.dream_1_tasks.widowx import ARM_FILE, WidowxMapping
from utils.common import portal_config
from utils.mujoco_sim import build_model


class WidowxTests(unittest.TestCase):
    def setUp(self):
        self.cfg = portal_config(OperatorConfig, "test", "widowx")
        self.mapping = WidowxMapping(self.cfg)

    def test_units_order_and_clamp(self):
        cfg = SimpleNamespace(state_schema=list(reversed(self.cfg.state_schema)),
                              action_schema=list(reversed(self.cfg.action_schema)), video_tracks=self.cfg.video_tracks)
        mapping = WidowxMapping(cfg)
        row = np.array([.1, .5, .6, .2, -.3, .4, .25, -.1, .6, .7, -.2, -.4, -.5, .75])
        state = dict(zip(mapping.fields, row))
        np.testing.assert_array_equal(mapping.to_model(state), row)
        self.assertEqual(mapping.to_robot(row), state)
        obs = mapping.observation(state, "move the cube")
        self.assertEqual(obs['state.left_arm'], row[:6].tolist())
        self.assertEqual(obs['state.right_arm'], row[7:13].tolist())
        self.assertEqual(obs['state.left_grip'], [.25])
        self.assertEqual(obs['state.right_grip'], [.75])
        self.assertEqual(mapping.to_robot(np.ones(14) * 100)['left_gripper.pos'], 1.)
        self.assertNotIn('scene_camera', mapping.cameras.values())
        with self.assertRaises(ValueError):
            WidowxMapping(portal_config(OperatorConfig, 'test', 'rebot'))
        with self.assertRaises(ValueError):
            mapping.to_robot(np.full(14, np.nan))

    def test_chunks_require_both_arms(self):
        left, right = np.ones((3, 7)), np.ones((3, 7)) * 2
        np.testing.assert_array_equal(self.mapping.chunk_rows({'q_l_chunk':left, 'q_r_chunk':right}),
                                      np.concatenate([left, right], axis=1))
        for reply in ({}, {'q_r_chunk':right}, {'q_l_chunk':left, 'q_r_chunk':right[:2]},
                      {'q_l_chunk':np.empty((0, 7)), 'q_r_chunk':np.empty((0, 7))},
                      {'q_l_chunk':left * np.nan, 'q_r_chunk':right}):
            with self.subTest(reply=repr(reply)), self.assertRaises(dream.DreamActionError):
                self.mapping.chunk_rows(reply)

    def test_dream_custom_arm_and_simulator_fk_agree(self):
        check_config(self.mapping.policy_config('dreamlabs/dream-1-beta', 20)['ik'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # TCP is measured from simulation, not hardware calibration.
            arm = arm_file.register(str(ARM_FILE))
        spec = EMBODIMENTS[arm].side_arm('left')
        urdf = yourdfpy.URDF.load(str(spec.urdf), load_meshes=False)
        model = build_model()
        data = mujoco.MjData(model)
        config = json.loads(ARM_FILE.read_text())
        for q in (config['rest'], [.2, .6, .8, .3, -.5, -.2], [-.4, .2, .3, -.4, .1, .6]):
            urdf.update_cfg(dict(zip(spec.joints, np.asarray(q, dtype=float))))
            transform = urdf.get_transform(config['flange'], config['base'])
            for side in ('left', 'right'):
                for joint, value in zip(spec.joints, q):
                    data.qpos[model.joint(f'{side}/{joint}').qposadr[0]] = value
                mujoco.mj_forward(model, data)
                flange = data.body(f'{side}/link_6')
                base = data.body(f'{side}/base_link')
                np.testing.assert_allclose(flange.xpos - base.xpos, transform[:3, 3], atol=1e-7)
                np.testing.assert_allclose(flange.xmat.reshape(3, 3), transform[:3, :3], atol=1e-7)
                tcp = flange.xpos + flange.xmat.reshape(3, 3) @ np.array(config['tcp'])
                np.testing.assert_allclose(tcp, data.site(f'{side}/ee_site').xpos, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
