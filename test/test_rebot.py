"""Single-arm wire mapping and hardware safety, without a robot or API key."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dream
import numpy as np
from livekit.portal import ObserverConfig, OperatorConfig, RobotConfig

from policies.dream_1_tasks.rebot import CAMERAS, RebotMapping, chunk_rows
from utils.common import portal_config
from utils.rebot import Arm, Follower, LIMITS, fields_from_config

REST = [0.0, -0.4, -0.5, 0.0, 0.0, 0.0, 0.5]


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = portal_config(OperatorConfig, "test", "rebot")
        self.fields = fields_from_config(self.cfg)
        self.mapping = RebotMapping(self.cfg)

    def test_all_roles_share_one_arm_two_cameras(self):
        for cls in (RobotConfig, OperatorConfig, ObserverConfig):
            cfg = portal_config(cls, "test", "rebot")
            self.assertEqual([f.name for f in cfg.state_schema], self.fields)
            self.assertEqual(set(cfg.video_tracks), set(CAMERAS.values()))
        with patch.dict(os.environ, {"PORTAL_PROFILE": "rebot"}):
            cfg = portal_config(ObserverConfig, "test")
            self.assertEqual(len(cfg.state_schema), 7)
        self.assertEqual(len(portal_config(RobotConfig, "test", "metal").state_schema), 14)
        with self.assertRaisesRegex(ValueError, "unknown Portal profile"):
            portal_config(RobotConfig, "test", "typo")

    def test_units_round_trip_and_model_padding_stays_off_portal(self):
        state = dict(zip(self.fields, REST))
        np.testing.assert_allclose(self.mapping.to_model(state), REST)
        self.assertEqual(self.mapping.to_robot(REST), state)
        obs = self.mapping.observation(state, "pick the cube")
        self.assertEqual(obs["state.right_grip"], [0.5])
        self.assertEqual(obs["state.right_arm"], REST[:6])
        from dream_ik.specs import EMBODIMENTS
        self.assertEqual(obs["state.left_arm"], list(EMBODIMENTS["seeed_faas"].side_arm("left").rest_pose))
        self.assertEqual(obs["state.left_grip"], [0.0])
        self.assertTrue(all(key.startswith("right_") for key in self.mapping.to_robot(REST)))
        policy = self.mapping.policy_config("dreamlabs/dream-1-beta", 20)
        self.assertEqual(policy["ik"], {"arm": "seeed_faas", "head": "eef"})
        from dream.client_ik import check_config
        check_config(policy["ik"])
        self.assertEqual(set(policy["embodiment"]["video"]), {"overhead", "wrist_l", "wrist_r"})

    def test_limits_are_rebot_limits_and_nan_is_rejected(self):
        row = [-100.0] * 7
        action = self.mapping.to_robot(row)
        self.assertEqual(list(action.values()), [lo for lo, _ in LIMITS])
        self.assertLess(action[self.fields[2]], 0)  # ReBot elbow has the opposite range to Metal
        for bad in ([0.0] * 6, [float("nan")] * 7, [float("inf")] * 7):
            with self.assertRaises(ValueError):
                self.mapping.to_robot(bad)

    def test_only_right_chunk_is_used(self):
        right = np.tile(REST, (8, 1))
        np.testing.assert_equal(chunk_rows({"q_r_chunk": right, "q_l_chunk": "ignored"}), right)
        for bad in ({}, {"q_r_chunk": None}, {"q_r_chunk": np.zeros((0, 7))},
                    {"q_r_chunk": np.zeros((8, 14))}, {"q_r_chunk": np.full((8, 7), np.nan)}):
            with self.assertRaises(dream.DreamActionError):
                chunk_rows(bad)


class FollowerTests(unittest.TestCase):
    def setUp(self):
        self.fields = fields_from_config(portal_config(RobotConfig, "test", "rebot"))
        self.arm = Mock(rest=REST[:6], errors=0)
        self.arm.read.return_value = list(REST)
        self.follower = Follower(self.arm, {}, self.fields, 0.03)
        self.follower.parked = False

    def test_failed_read_prevents_action_even_after_a_good_read(self):
        state, _ = self.follower.get_observation()
        self.follower.send_action(state)
        self.arm.command.assert_called_once()
        self.arm.command.reset_mock()
        self.arm.read.return_value = None
        self.assertEqual(self.follower.get_observation(), ({}, {}))
        self.follower.send_action(state)
        self.arm.command.assert_not_called()

    def test_target_ramp_and_parked_hold(self):
        state, _ = self.follower.get_observation()
        state[self.fields[0]] = 2.0
        self.follower.send_action(state)
        self.assertAlmostEqual(self.arm.command.call_args.args[0][0], 0.03)
        self.follower.parked = True
        self.follower.send_action(state)
        self.arm.command.assert_called_once()

    def test_incomplete_or_nonfinite_action_never_reaches_motor(self):
        state, _ = self.follower.get_observation()
        for bad in ({}, {**state, "left_gripper.pos": 1.0}, {**state, self.fields[0]: float("nan")}):
            with self.assertRaises(ValueError):
                self.follower.send_action(bad)
        self.arm.command.assert_not_called()

    def test_park_verifies_rest_before_disabling(self):
        with patch("utils.metal.time.sleep"):
            self.follower.park(1)
        self.arm.switch_off.assert_called_once()
        self.arm.connect.assert_called_once_with(False)
        self.assertTrue(self.follower.parked)

    def test_failed_home_leaves_power_on(self):
        self.arm.read.return_value = None
        with self.assertRaises(RuntimeError):
            self.follower.park(1)
        self.arm.switch_off.assert_not_called()
        self.assertFalse(self.follower.parked)

    def test_blocked_park_keeps_power(self):
        self.arm.read.return_value = [1.0, *REST[1:]]
        with patch("utils.metal.time.sleep"), self.assertRaisesRegex(RuntimeError, "behind"):
            self.follower.park(1)
        self.arm.switch_off.assert_not_called()

    def test_read_lost_during_park_stops_the_walk(self):
        self.arm.read.side_effect = [list(REST), None]
        with patch("utils.metal.time.sleep"), self.assertRaisesRegex(RuntimeError, "unreadable"):
            self.follower.park(1)
        self.arm.command.assert_called_once()
        self.arm.switch_off.assert_not_called()

    def test_failed_connect_cleanup_is_safe_to_repeat(self):
        camera = Mock(is_connected=False, thread=None)
        camera.connect.side_effect = lambda: setattr(camera, "is_connected", True)
        camera.disconnect.side_effect = lambda: setattr(camera, "is_connected", False)
        self.follower.cameras = {"overhead_camera": SimpleNamespace(_cam=camera)}
        self.arm.connect.side_effect = RuntimeError("startup outside limits")
        with self.assertRaisesRegex(RuntimeError, "outside limits"):
            self.follower.connect()
        self.follower.disconnect()  # robot's finally block
        camera.disconnect.assert_called_once()
        self.arm.switch_off.assert_not_called()


class MotorBridgeTests(unittest.TestCase):
    def setUp(self):
        self.arm = Arm("/dev/test", "dm-serial", 0.0, -5.0)
        self.arm.ctrl = Mock()
        self.arm.motors = [Mock() for _ in range(7)]
        for i, motor in enumerate(self.arm.motors, 1):
            motor.get_register_u32.return_value = i
            motor.get_register_f32.return_value = -2.5 if i == 7 else REST[i-1]
            motor.get_state.return_value = SimpleNamespace(pos=-2.5 if i == 7 else REST[i-1], status_code=0)

    def test_reads_and_commands_use_normalized_gripper(self):
        self.assertEqual(self.arm.read(), REST)
        self.arm.command(REST)
        self.arm.motors[-1].send_pos_vel.assert_called_once_with(pos=-2.5, vlim=0.5)

    def test_missing_motor_rejects_whole_state_and_keeps_power(self):
        self.arm.motors[2].get_register_f32.side_effect = TimeoutError("silent motor")
        self.assertIsNone(self.arm.read())
        self.arm.ctrl.disable_all.assert_not_called()

    def test_position_comes_from_a_fresh_register_read(self):
        self.arm.motors[0].get_state.return_value.pos = 100.0
        self.assertEqual(self.arm.read(), REST)
        self.arm.motors[0].get_register_f32.assert_called_once_with(80, timeout_ms=100)

    def test_fault_or_disabled_motor_is_not_a_valid_powered_state(self):
        self.arm.energized = True
        self.assertIsNone(self.arm.read())
        self.arm.energized = False
        self.arm.motors[2].get_state.return_value.status_code = 8
        self.assertIsNone(self.arm.read())

    def test_close_never_disables(self):
        ctrl = self.arm.ctrl
        self.arm.close()
        ctrl.close_bus.assert_called_once()
        ctrl.close.assert_called_once()
        ctrl.disable_all.assert_not_called()
        ctrl.shutdown.assert_not_called()

    def driver_module(self):
        ctrl = self.arm.ctrl
        ctrl.add_damiao_motor.side_effect = self.arm.motors
        module = SimpleNamespace(Controller=Mock(), Mode=SimpleNamespace(POS_VEL=2))
        module.Controller.from_dm_serial.return_value = ctrl
        return module, ctrl

    def test_connect_seeds_before_enabling_and_uses_dm_models(self):
        module, ctrl = self.driver_module()
        motors = list(self.arm.motors)
        events = []
        for motor in motors:
            motor.send_pos_vel.side_effect = lambda **kwargs: events.append("target")
        ctrl.enable_all.side_effect = lambda: events.append("enable")
        with patch.dict("sys.modules", {"motorbridge": module}):
            self.arm.connect()
        self.assertEqual(events, ["target"] * 7 + ["enable"] + ["target"] * 7)
        self.assertEqual(self.arm.rest, REST[:6])
        self.assertEqual([call.args for call in ctrl.add_damiao_motor.call_args_list],
                         [(i, i + 0x10, "4340P" if i <= 3 else "4310") for i in range(1, 8)])
        for motor in motors:
            motor.ensure_mode.assert_called_once_with(2, timeout_ms=1000)

    def test_partial_enable_failure_never_disables(self):
        module, ctrl = self.driver_module()
        ctrl.enable_all.side_effect = OSError("enable failed halfway through")
        with patch.dict("sys.modules", {"motorbridge": module}), self.assertRaises(OSError):
            self.arm.connect()
        ctrl.disable_all.assert_not_called()
        ctrl.close.assert_called_once()

    def test_small_startup_settling_clamps_home_and_all_targets(self):
        module, ctrl = self.driver_module()
        motor = self.arm.motors[2]
        motor.get_register_f32.return_value = 0.004669189453125
        with patch.dict("sys.modules", {"motorbridge": module}):
            self.arm.connect()
        self.assertEqual(self.arm.rest[2], 0.0)
        self.assertEqual(motor.send_pos_vel.call_count, 2)
        for call in motor.send_pos_vel.call_args_list:
            self.assertEqual(call.kwargs["pos"], 0.0)
        ctrl.enable_all.assert_called_once()

    def test_startup_outside_margin_identifies_joint_and_never_writes(self):
        for joint, position in ((3, 0.02), (1, -2.62)):
            with self.subTest(joint=joint, position=position):
                self.setUp()
                module, ctrl = self.driver_module()
                motors = list(self.arm.motors)
                motors[joint - 1].get_register_f32.return_value = position
                with patch.dict("sys.modules", {"motorbridge": module}), self.assertRaisesRegex(
                        RuntimeError, f"joint{joint}=.*limit"):
                    self.arm.connect()
                ctrl.enable_all.assert_not_called()
                ctrl.disable_all.assert_not_called()
                for motor in motors:
                    motor.send_pos_vel.assert_not_called()
                    motor.ensure_mode.assert_not_called()
                    motor.write_register_f32.assert_not_called()

    def test_raw_gripper_outside_calibration_never_enables(self):
        for position in (1.868, -5.1):
            with self.subTest(position=position):
                self.setUp()
                module, ctrl = self.driver_module()
                motors = list(self.arm.motors)
                motors[-1].get_register_f32.return_value = position
                with patch.dict("sys.modules", {"motorbridge": module}), self.assertRaisesRegex(
                        RuntimeError, "gripper=.*calibrated range"):
                    self.arm.connect()
                ctrl.enable_all.assert_not_called()
                for motor in motors:
                    motor.send_pos_vel.assert_not_called()
                    motor.ensure_mode.assert_not_called()
                    motor.write_register_f32.assert_not_called()

    def test_bad_startup_read_never_enables(self):
        module, ctrl = self.driver_module()
        self.arm.motors[2].get_register_u32.side_effect = TimeoutError()
        with patch.dict("sys.modules", {"motorbridge": module}), self.assertRaises(TimeoutError):
            self.arm.connect()
        ctrl.enable_all.assert_not_called()
        ctrl.disable_all.assert_not_called()


if __name__ == "__main__":
    unittest.main()
