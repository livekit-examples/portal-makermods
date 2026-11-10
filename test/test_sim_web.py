"""Same-origin access and viewer-token permissions without a LiveKit server."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jwt
from aiohttp.test_utils import TestClient, TestServer
from livekit.portal import RobotConfig
from utils.common import portal_config
from utils.sim_web import BrowserServer, viewer_token


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = BrowserServer("test", portal_config(RobotConfig, "test", "widowx"), {},
                                    SimpleNamespace(), SimpleNamespace(), "test-access-code")
        self.client = TestClient(TestServer(self.server.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_unauthenticated_requests_and_cross_origin_login_rejected(self):
        self.assertEqual((await self.client.get('/api/session')).status, 401)
        self.assertEqual((await self.client.get('/ws')).status, 401)
        response = await self.client.post('/api/login', json={'code':'test-access-code'},
                                          headers={'Origin':'https://untrusted.example'})
        self.assertEqual(response.status, 403)
        response = await self.client.post('/api/login', json={'code':'wrong'})
        self.assertEqual(response.status, 401)

    async def test_login_cookie_and_expiry(self):
        response = await self.client.post('/api/login', json={'code':'test-access-code'})
        self.assertEqual(response.status, 200)
        self.assertTrue(response.cookies['sim_session']['httponly'])
        self.assertEqual(response.cookies['sim_session']['samesite'], 'Strict')
        self.assertEqual((await self.client.get('/api/session')).status, 200)
        self.server.sessions.clear()
        self.assertEqual((await self.client.get('/api/session')).status, 401)

    async def test_readiness_waits_for_first_simulation_tick(self):
        self.assertEqual((await self.client.get('/healthz')).status, 503)
        self.server.status = {'episode':1}
        self.assertEqual((await self.client.get('/healthz')).status, 200)

    def test_video_token_cannot_publish_or_control(self):
        with patch.dict('os.environ', LIVEKIT_API_KEY='devkey', LIVEKIT_API_SECRET='s' * 32):
            data = jwt.decode(viewer_token('viewer', 'sim-room'), options={'verify_signature':False})
        self.assertEqual(data['video']['room'], 'sim-room')
        self.assertFalse(data['video']['canPublish'])
        self.assertFalse(data['video']['canPublishData'])
        self.assertTrue(data['video']['canSubscribe'])


if __name__ == '__main__':
    unittest.main()
