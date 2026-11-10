"""Recorder profile selection and single/dual-arm live layouts without hardware."""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import recorder
from livekit.portal import ObserverConfig
from utils.common import portal_config


class RecorderProfiles(unittest.IsolatedAsyncioTestCase):
    async def start_recorder(self, profile, *, environment='rebot', observe_only=False, fail_sink=False):
        observer = Mock(connect=AsyncMock(), disconnect=AsyncMock())
        if fail_sink:
            observer.record_to.side_effect = RuntimeError('cannot open archive')
        with (patch.dict(os.environ, LIVEKIT_URL='ws://test', LIVEKIT_ROOM='test', PORTAL_PROFILE=environment),
              patch.object(recorder, 'load_env'), patch.object(recorder, 'mint_token', return_value='test-token'),
              patch.object(recorder, 'Observer', return_value=observer) as create,
              patch.object(recorder, 'RrdSink'), patch.object(recorder, 'LiveSink'),
              patch.object(recorder.asyncio, 'sleep', AsyncMock(side_effect=asyncio.CancelledError))):
            if fail_sink:
                with self.assertRaisesRegex(RuntimeError, 'cannot open archive'):
                    await recorder.main(observe_only, profile)
            else:
                await recorder.main(observe_only, profile)
        observer.disconnect.assert_awaited_once()
        observer.close.assert_called_once()
        return create.call_args.args[0]

    async def test_explicit_trossen_overrides_rebot_environment(self):
        cfg = await self.start_recorder('widowx')
        self.assertEqual(len(cfg.state_schema), 14)
        self.assertEqual(len(cfg.action_schema), 14)
        self.assertEqual(set(cfg.video_tracks), {'overhead_camera', 'left_wrist_camera',
                                                'right_wrist_camera', 'scene_camera'})

    async def test_yam_profile_overrides_environment(self):
        cfg = await self.start_recorder('yam', environment='widowx')
        names = {f.name for f in cfg.state_schema}
        self.assertEqual(len(names), 14)
        self.assertIn('left_joint_6.pos', names)
        self.assertNotIn('left_joint_0.pos', names)
        self.assertEqual(len(cfg.video_tracks), 4)

    async def test_rebot_flag_and_environment_remain_supported(self):
        for profile, environment in [('rebot', 'widowx'), (None, 'rebot')]:
            cfg = await self.start_recorder(profile, environment=environment)
            self.assertEqual(len(cfg.state_schema), 7)
            self.assertTrue(all(f.name.startswith('right_') for f in cfg.state_schema))
            self.assertEqual(set(cfg.video_tracks), {'overhead_camera', 'arm_camera'})

    async def test_observe_only_uses_selected_profile_and_failure_closes_observer(self):
        cfg = await self.start_recorder('widowx', observe_only=True)
        self.assertEqual(len(cfg.video_tracks), 4)
        await self.start_recorder('rebot', fail_sink=True)

    def test_layout_only_shows_existing_arms_and_all_cameras(self):
        for profile, sides in [('widowx', {'left', 'right'}), ('yam', {'left', 'right'}), ('rebot', {'right'}), ('metal', {'left', 'right'})]:
            cfg = portal_config(ObserverConfig, 'test', profile)
            session = SimpleNamespace(state_schema=cfg.state_schema,
                                      video_tracks=[SimpleNamespace(name=name) for name in cfg.video_tracks])
            with self.subTest(profile=profile), patch.object(recorder, 'rrb') as blueprints:
                recorder.layout(session)
                titles = {call.kwargs['name'] for call in blueprints.TimeSeriesView.call_args_list}
                self.assertEqual({side for side in ('left', 'right') if f'{side} joints (rad)' in titles}, sides)
                self.assertEqual({call.kwargs['name'] for call in blueprints.Spatial2DView.call_args_list},
                                 set(cfg.video_tracks))


if __name__ == '__main__':
    unittest.main()
