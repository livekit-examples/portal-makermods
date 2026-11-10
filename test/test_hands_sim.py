"""Synthetic webcam gestures through the actual simulation IK and Portal contracts."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from livekit.portal import OperatorConfig, RobotConfig
from hand_teleop import Sighting, build_pilot, request_home_or_park
from utils.common import portal_config
from utils.hands import ArmsIK, Hand
from utils.mujoco_sim import Simulation
from test.test_hands import hand_world, project, rot
from utils.hands import palm_rotation, turn_between


class SimHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iks = {p: ArmsIK(30, 0., .5, profile=p) for p in ('widowx', 'yam')}
        for ik in cls.iks.values():
            ik.warmup()

    @staticmethod
    def sight(pilot, seq, *, side='left', pinch=.054, position=(0., 0., .5), rotation=np.eye(3)):
        now = 100. + seq / 30
        hand = project(hand_world(pinch_m=pinch), np.array(position), rotation)
        pilot.see(Sighting(seq, now, np.zeros((480, 640, 3), np.uint8),
                           [Hand(side, hand.image, hand.world)]), now)

    def test_gestures_normalized_grip_hold_and_retake(self):
        for profile, ik in self.iks.items():
            with self.subTest(profile=profile):
                pilot = build_pilot(30, ik, 'off', 640)
                state = {name: value for side in ik.sides
                         for name, value in zip(ik.fields[side], [*ik.rest, .4])}
                for seq in range(1, 4):self.sight(pilot, seq)
                self.assertIsNone(pilot.tick(False, state))
                self.assertIsNone(pilot.tick(True, {'left_gripper.pos': .4}))
                action = pilot.tick(True, state)
                cfg = portal_config(OperatorConfig, 'test', profile)
                self.assertEqual(set(action), {f.name for f in cfg.action_schema})
                self.assertAlmostEqual(action['left_gripper.pos'], .4, places=5)
                for seq in range(4, 64):
                    self.sight(pilot, seq, pinch=.09, position=(0., -.025, .5))
                    old = action
                    action = pilot.tick(True, state)
                    for name in ik.fields['left'][:6]:
                        self.assertLessEqual(abs(action[name] - old[name]), .5 / 30 + 1e-6)
                self.assertGreater(action['left_gripper.pos'], .95)
                self.assertLessEqual(action['left_gripper.pos'], 1.)
                self.assertGreater(np.linalg.norm([action[n]-state[n] for n in ik.fields['left'][:6]]), .01)
                for name in ik.fields['right']:
                    self.assertEqual(action[name], state[name], 'inactive arm must hold its own state')
                pilot.drop()
                moved = dict(action)
                moved[ik.fields['left'][0]] = .2
                retaken = pilot.tick(True, moved)
                self.assertAlmostEqual(retaken[ik.fields['left'][0]], .2, delta=.5/30)
                pilot.see(None, 1000.)
                self.assertIsNone(pilot.tick(True, moved), 'lost webcam must release the clutch')

    def test_fk_limits_and_commands_match_simulator(self):
        for profile, ik in self.iks.items():
            sim = Simulation(portal_config(RobotConfig, 'test', profile), arms=profile, render=False)
            try:
                for side in ik.sides:
                    q = np.array([sim.state()[n] for n in ik.fields[side][:6]])
                    tcp = ik.fk(q, side)
                    origin = np.array([-.28, .3 if side == 'left' else -.3, 0])
                    np.testing.assert_allclose(tcp[:3, 3] + origin,
                                               sim.data.site(f'{side}/' + ('ee_site' if profile == 'widowx' else 'grasp_site')).xpos, atol=1e-6)
                self.assertEqual(ik.clamp_action(sim.state()), sim.validate_action(sim.state()))
                q = {side: ik.rest.copy() for side in ik.sides}
                goals = {side: ik.fk(q[side], side) for side in ik.sides}
                goals['right'][2, 3] = -.2
                for _ in range(90):q, _ = ik.step(goals, q)
                self.assertGreater(ik.fk(q['right'], 'right')[2, 3], 0.)
            finally:sim.close()

    def test_tool_pov_roll_and_depth_with_both_rigs_and_sides(self):
        palm = palm_rotation(hand_world())
        for profile, ik in self.iks.items():
            for side in ik.sides:
                with self.subTest(profile=profile, side=side):
                    pilot = build_pilot(30, ik, 'full', 640, frame='tool')
                    state = {name: value for s in ik.sides
                             for name, value in zip(ik.fields[s], [*ik.rest, .4])}
                    # Hold the hand sideways before engaging: no absolute orientation is imposed.
                    held = rot('z', 65)
                    for seq in range(1, 4):self.sight(pilot, seq, side=side, rotation=held)
                    action = pilot.tick(True, state)
                    start = ik.fk(pilot.q[side], side)
                    for seq in range(4, 94):
                        angle = 25 * min(1., (seq - 3) / 30)
                        wrist = held @ palm @ rot('x', angle) @ palm.T
                        self.sight(pilot, seq, side=side, rotation=wrist)
                        previous = action
                        action = pilot.tick(True, state)
                        for name in ik.fields[side][:6]:
                            self.assertLessEqual(abs(action[name] - previous[name]), .5 / 30 + 1e-6)
                    turned = ik.fk(pilot.q[side], side)
                    expected = start[:3, :3] @ rot('z', 25 - 5 - 1.5)
                    self.assertLess(turn_between(expected, turned[:3, :3]), np.radians(3))
                    self.assertLess(np.linalg.norm(turned[:3, 3] - start[:3, 3]), .003)
                    for seq in range(94, 184):
                        depth = .5 - .02 * min(1., (seq - 93) / 30)
                        self.sight(pilot, seq, side=side, position=(0., 0., depth), rotation=wrist)
                        action = pilot.tick(True, state)
                    moved = ik.fk(pilot.q[side], side)[:3, 3] - turned[:3, 3]
                    forward = turned[:3, 2]
                    self.assertGreater(moved @ forward, .004)
                    self.assertLess(np.linalg.norm(moved - (moved @ forward) * forward), .003)
                    other = 'right' if side == 'left' else 'left'
                    for name in ik.fields[other]:self.assertEqual(action[name], state[name])

    def test_schema_validation_does_not_depend_on_yaml_order(self):
        for profile in self.iks:
            cfg = portal_config(OperatorConfig, 'test', profile)
            shuffled = SimpleNamespace(state_schema=list(reversed(cfg.state_schema)),
                                       action_schema=list(reversed(cfg.action_schema)))
            ik = ArmsIK(30, 0., .5, profile=profile, cfg=shuffled)
            self.assertEqual(ik.fields, self.iks[profile].fields)
            wrong = portal_config(OperatorConfig, 'test', 'yam' if profile == 'widowx' else 'widowx')
            with self.assertRaises(ValueError):ArmsIK(30, 0., .5, profile=profile, cfg=wrong)


class MotionKeysTests(unittest.IsolatedAsyncioTestCase):
    async def test_sim_home_and_hardware_park(self):
        for profile in ('widowx', 'yam', 'metal', 'rebot'):
            op = SimpleNamespace(perform_rpc=AsyncMock(return_value='queued'))
            with patch('hand_teleop.request_park', new_callable=AsyncMock) as park:
                await request_home_or_park(op, profile)
                if profile in ('widowx', 'yam'):
                    op.perform_rpc.assert_awaited_once_with('sim.motion', 'home', destination='robot')
                    park.assert_not_awaited()
                else:
                    park.assert_awaited_once_with(op, 'hands')
                    op.perform_rpc.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
