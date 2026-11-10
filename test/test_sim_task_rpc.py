"""Real browser WebSocket → Portal RPC → Dream TaskControl; no inference or GPU.

With a disposable `livekit-server --dev`:
SIM_TASK_TEST_LIVEKIT=ws://127.0.0.1:7880 python -m unittest test.test_sim_task_rpc
"""
import asyncio
import os
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from livekit.portal import Operator, OperatorConfig, Robot, RobotConfig
from policies.dream_1_tasks.run import TaskControl
from utils.common import DREAM_TASKS_IDENTITY, mint_token, portal_config
from utils.sim_web import BrowserServer


@unittest.skipUnless(os.environ.get('SIM_TASK_TEST_LIVEKIT'), 'requires a disposable LiveKit dev server')
class TaskRpcTests(unittest.IsolatedAsyncioTestCase):
    async def test_widowx_task_delivery_and_handoff(self):
        await self.check_profile('widowx')

    async def test_yam_task_delivery_and_handoff(self):
        await self.check_profile('yam')

    async def check_profile(self, profile):
        # Portal's UniFFI callbacks use call_soon across threads; use the normal app loop mode.
        asyncio.get_running_loop().set_debug(False)
        url = os.environ['SIM_TASK_TEST_LIVEKIT']
        room = 'sim-task-test-' + uuid.uuid4().hex[:10]
        with patch.dict(os.environ, LIVEKIT_URL=url, LIVEKIT_API_KEY='devkey', LIVEKIT_API_SECRET='secret'):
            cfg = portal_config(RobotConfig, room, profile)
            robot = Robot(cfg)
            policy = Operator(portal_config(OperatorConfig, room, profile))
            control = TaskControl(policy, [], profile)
            control.register_rpcs()
            browser = BrowserServer(room, cfg, {}, robot, SimpleNamespace(busy=False), 'test-code', profile=profile)
            robot.on_operator_joined(browser.tasks.connection_changed)
            robot.on_operator_left(browser.tasks.connection_changed)
            client = TestClient(TestServer(browser.app))
            try:
                await robot.connect(url, mint_token('robot', room))
                await client.start_server()
                browser.task_sync = asyncio.create_task(browser.tasks.run())
                self.assertEqual((await client.post('/api/login', json={'code':'test-code'})).status, 200)
                ws = await client.ws_connect('/ws')
                self.assertEqual((await ws.receive_json(timeout=15))['type'], 'ready')

                async def until(predicate):
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        state = await ws.receive_json(timeout=5)
                        self.assertNotEqual(state['type'], 'error', state)
                        if state['type'] == 'state' and predicate(state):
                            return state
                    self.fail('timed out waiting for browser state')

                await ws.send_json({'type':'task', 'task':'Put the cube in the tray'})
                await until(lambda s: s['dream_task']['task'] == 'Put the cube in the tray')
                self.assertIsNone(control.task)
                await policy.connect(url, mint_token(DREAM_TASKS_IDENTITY, room))
                await until(lambda s: s['dream_task']['applied'])
                self.assertEqual(control.task, 'Put the cube in the tray')
                self.assertIsNone(robot.active_operator())

                await ws.send_json({'type':'control', 'identity':DREAM_TASKS_IDENTITY,
                                    'task':'Open both grippers'})
                await until(lambda s: s['active'] == DREAM_TASKS_IDENTITY)
                self.assertEqual(control.task, 'Open both grippers')
                await ws.send_json({'type':'task', 'task':'Move the cube back'})
                await until(lambda s: s['dream_task']['applied'] and s['active'] is None
                            and s['dream_task']['task'] == 'Move the cube back')
                self.assertEqual(control.task, 'Move the cube back')

                await policy.disconnect()
                await until(lambda s: DREAM_TASKS_IDENTITY not in s['operators'])
                control.task = None
                await policy.connect(url, mint_token(DREAM_TASKS_IDENTITY, room))
                await until(lambda s: s['dream_task']['applied'])
                self.assertEqual(control.task, 'Move the cube back')
                self.assertIsNone(robot.active_operator())
                await ws.close()
            finally:
                await client.close()
                await policy.disconnect()
                policy.close()
                await robot.disconnect()
                robot.close()


if __name__ == '__main__':
    unittest.main()
