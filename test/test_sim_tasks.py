"""Browser instruction synchronization against the real TaskControl, without inference."""
import asyncio
import json
import unittest

from policies.dream_1_tasks.run import TaskControl
from test.test_dream1_tasks import FakeOperator
from utils.common import DREAM_TASKS_IDENTITY as ME, DREAM_RESUME_RPC, DREAM_SET_TASK_RPC, DREAM_STATUS_RPC
from utils.sim_tasks import DreamTaskSync, task_text


class Peer:
    def __init__(self):
        self.op = FakeOperator()
        self.tasks = TaskControl(self.op, [], 'widowx')
        self.connected = False
        self.calls = []
        self.fail = False

    def operators(self):
        return [ME] if self.connected else []

    async def perform_rpc(self, method, payload, **kwargs):
        self.calls.append(method)
        if self.fail:
            raise RuntimeError('temporary failure')
        if method == DREAM_STATUS_RPC:
            result = self.tasks.status()
        elif method == DREAM_SET_TASK_RPC:
            result = await self.tasks.set_task(json.loads(payload)['task'], 'browser')
        elif method == DREAM_RESUME_RPC:
            result = await self.tasks.resume('browser')
        else:
            raise AssertionError(method)
        return json.dumps(result)


class SyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.peer = Peer()
        self.sync = DreamTaskSync(self.peer)

    async def test_late_connection_edit_handoff_and_reconnect(self):
        await self.sync.set_task('  move  the cube ')
        await self.sync.tick()
        self.assertFalse(self.peer.calls)
        self.assertIn('queued', self.sync.message)
        self.peer.connected = True
        await self.sync.tick()
        self.assertEqual(self.peer.tasks.task, 'move the cube')
        self.assertIsNone(self.peer.op.active)
        await self.sync.handoff('move the cube')
        self.assertEqual(self.peer.op.active, ME)
        count = len(self.peer.calls)
        await self.sync.tick()
        self.assertEqual(len(self.peer.calls), count)  # no repeated task RPC pausing a running policy
        await self.sync.set_task('open the grippers')
        await self.sync.tick()
        self.assertIsNone(self.peer.op.active)
        self.assertEqual(self.peer.tasks.task, 'open the grippers')
        self.peer.tasks.task = None
        self.sync.connection_changed(ME)  # even a quick reconnect between polling ticks
        await self.sync.tick()
        self.assertEqual(self.peer.tasks.task, 'open the grippers')
        self.assertIsNone(self.peer.op.active)

    async def test_same_task_does_not_interrupt_and_human_keeps_control(self):
        self.peer.connected = True
        self.peer.op.active = 'browser-human'
        await self.sync.set_task('move cube')
        await self.sync.tick()
        self.assertEqual(self.peer.op.active, 'browser-human')
        self.peer.op.active = ME
        self.sync.connection_changed(ME)
        await self.sync.tick()
        self.assertEqual(self.peer.op.active, ME)

    async def test_failures_retry_and_never_resume(self):
        self.peer.connected = True
        self.peer.fail = True
        await self.sync.set_task('move cube')
        await self.sync.tick()
        self.assertFalse(self.sync.applied)
        self.assertIn('temporary failure', self.sync.message)
        with self.assertRaises(RuntimeError):
            await self.sync.handoff('move cube')
        self.assertNotIn(DREAM_RESUME_RPC, self.peer.calls)
        self.peer.fail = False
        await self.sync.tick()
        self.assertTrue(self.sync.applied)
        self.assertIsNone(self.peer.op.active)

    async def test_wrong_embodiment_rejected(self):
        self.peer.connected = True
        self.peer.tasks.profile = 'rebot'
        with self.assertRaisesRegex(ValueError, '--profile widowx'):
            await self.sync.handoff('move cube')
        self.assertNotIn(DREAM_SET_TASK_RPC, self.peer.calls)
        self.assertNotIn(DREAM_RESUME_RPC, self.peer.calls)

    async def test_yam_task_sync_requires_matching_policy(self):
        self.sync = DreamTaskSync(self.peer, "yam")
        self.peer.connected = True
        with self.assertRaisesRegex(ValueError, '--profile yam'):
            await self.sync.handoff('unplug the cable')
        self.assertNotIn(DREAM_SET_TASK_RPC, self.peer.calls)
        self.assertNotIn(DREAM_RESUME_RPC, self.peer.calls)
        self.peer.tasks.profile = 'yam'
        await self.sync.tick()
        self.assertEqual(self.peer.tasks.task, 'unplug the cable')
        self.assertIsNone(self.peer.op.active)
        await self.sync.handoff('unplug the cable')
        self.assertEqual(self.peer.op.active, ME)

    async def test_concurrent_edit_cannot_start_the_wrong_task(self):
        self.peer.connected = True
        await asyncio.gather(self.sync.handoff('first task'), self.sync.set_task('second task'))
        await self.sync.tick()
        self.assertEqual(self.peer.tasks.task, 'second task')
        self.assertIsNone(self.peer.op.active)

    def test_validation(self):
        for value in ('', ' \n ', 'x' * 201, None, 42):
            with self.subTest(value=value), self.assertRaises(ValueError):
                task_text(value)


if __name__ == '__main__':
    unittest.main()
