"""Process boundary: deterministic commands, errors, and cleanup without a GPU."""
import unittest

from livekit.portal import RobotConfig
from utils.common import portal_config
from utils.mujoco_sim import Simulation
from utils.sim_process import SimulationProcess


class SimulationProcessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = portal_config(RobotConfig, 'process-test', 'yam')

    def test_roundtrip_matches_local_physics_and_closes(self):
        remote = SimulationProcess(self.cfg, arms='yam', render=False)
        local = Simulation(self.cfg, arms='yam', render=False)
        try:
            action = local.state() | {'left_joint_1.pos': .3}
            self.assertEqual(remote.validate_action(action), local.validate_action(action))
            for _ in range(3):
                state, frames, status = remote.tick(action=action)
                expected, _, local_status = local.tick(action=action)
                self.assertEqual(state, expected)
                self.assertEqual(status, local_status)
                self.assertEqual(frames, {})
            with self.assertRaises(ValueError):
                remote.validate_action({})
        finally:
            remote.close()
            remote.close()  # repeated cleanup is harmless
            local.close()
        self.assertFalse(remote.process.is_alive())
        self.assertEqual(remote.process.exitcode, 0)

    def test_worker_error_is_reported_and_stops_worker(self):
        remote = SimulationProcess(self.cfg, arms='yam', render=False)
        try:
            with self.assertRaisesRegex(RuntimeError, 'unknown scene setup'):
                remote.tick(reset=True, setup='invalid')
        finally:
            remote.close()
        self.assertFalse(remote.process.is_alive())

    def test_startup_error_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, 'unknown renderer'):
            SimulationProcess(self.cfg, arms='yam', render=False, renderer='invalid')


if __name__ == '__main__':
    unittest.main()
