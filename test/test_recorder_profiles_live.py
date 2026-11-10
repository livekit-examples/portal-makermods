"""Round-trip supported simulation and ReBot profiles through the actual recorder, no hardware or Dream API.

RECORDER_TEST_LIVEKIT=ws://127.0.0.1:7880 python -m unittest test.test_recorder_profiles_live
Requires a disposable recent `livekit-server --dev` with frame timestamps.
"""
import asyncio
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

import recorder
from livekit.portal import Operator, OperatorConfig, Robot, RobotConfig
from livekit.portal.convert import EpisodeRule, Selection, read_archive
from test._harness import frame_for_tick, tick_from_frame, TICK_WRAP
from utils.common import mint_token, portal_config, pace


@unittest.skipUnless(os.environ.get('RECORDER_TEST_LIVEKIT'), 'requires a disposable LiveKit dev server')
class RecorderRoundTrip(unittest.IsolatedAsyncioTestCase):
    async def test_dual_simulation_and_single_rebot(self):
        # Match the app's loop mode; Portal's native callbacks are not asyncio-debug compatible.
        asyncio.get_running_loop().set_debug(False)
        for profile, fields, cameras in [('widowx', 14, 4), ('yam', 14, 4), ('rebot', 7, 2)]:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory(prefix='portal-profile-rrd-') as directory:
                room = f'recorder-{profile}-{uuid.uuid4().hex[:8]}'
                url = os.environ['RECORDER_TEST_LIVEKIT']
                with (patch.dict(os.environ, LIVEKIT_URL=url, LIVEKIT_ROOM=room,
                                 LIVEKIT_API_KEY='devkey', LIVEKIT_API_SECRET='secret',
                                 PORTAL_DATA_RRD_DIR=directory, PORTAL_PROFILE='metal'),
                      patch.object(recorder, 'load_env')):
                    cfg = portal_config(RobotConfig, room, profile)
                    robot = Robot(cfg)
                    op = Operator(portal_config(OperatorConfig, room, profile))
                    latest = None

                    def observed(obs):
                        nonlocal latest
                        latest = obs

                    op.on_observation(observed)
                    robot.on_action(lambda _: None)
                    writer = publisher = None

                    async def publish():
                        async for tick in pace(cfg.fps):
                            timestamp = robot.now_us()
                            robot.send_state({f.name: float(tick % TICK_WRAP) for f in cfg.state_schema},
                                             timestamp_us=timestamp)
                            for camera in cfg.video_tracks:
                                robot.send_video_frame(camera, frame_for_tick(tick % TICK_WRAP),
                                                       timestamp_us=timestamp)

                    async def wait_for(predicate):
                        deadline = time.monotonic() + 15
                        while not predicate():
                            if writer and writer.done():
                                await writer
                                self.fail('recorder exited before recording')
                            self.assertLess(time.monotonic(), deadline, 'timed out waiting for streams')
                            await asyncio.sleep(.05)

                    try:
                        await robot.connect(url, mint_token('robot', room))
                        publisher = asyncio.create_task(publish())
                        writer = asyncio.create_task(recorder.main(False, profile))
                        await op.connect(url, mint_token('profile-test-human', room))
                        await wait_for(lambda: latest is not None and bool(list(Path(directory).glob('*.rrd'))))
                        await asyncio.sleep(.5)
                        await op.set_active_operator('profile-test-human')
                        await wait_for(lambda: op.active_operator() == 'profile-test-human')
                        heard = await op.send_keypoint('recording', {'task_description':f'{profile} test'},
                                                       timestamp_us=latest.timestamp_us)
                        self.assertIn('recorder', heard)
                        for _ in range(45):
                            obs = latest
                            op.send_action(dict(obs.state), in_reply_to_ts_us=obs.timestamp_us)
                            await asyncio.sleep(1 / cfg.fps)
                        await op.send_keypoint('idle', {'task_description':f'{profile} test'},
                                               timestamp_us=latest.timestamp_us)
                        await asyncio.sleep(.3)
                    finally:
                        if writer:
                            writer.cancel()
                            await writer  # cancellation must flush and close RrdSink
                        if publisher:
                            publisher.cancel()
                            await asyncio.gather(publisher, return_exceptions=True)
                        await op.disconnect()
                        op.close()
                        await robot.disconnect()
                        robot.close()

                    paths = list(Path(directory).glob('*.rrd'))
                    self.assertEqual(len(paths), 1)
                    archive = read_archive(paths[0])
                    episodes = archive.episodes(EpisodeRule())
                    self.assertEqual([episode.task for episode in episodes], [f'{profile} test'])
                    rows = list(archive.rows(episodes[0], Selection(max_stale_us=0)))
                    self.assertGreater(len(rows), 20)
                    self.assertEqual(set(archive.schema.cameras), set(cfg.video_tracks))
                    for row in rows:
                        self.assertEqual(len(row.state), fields)
                        self.assertEqual(len(row.action), fields)
                        self.assertEqual(len(row.frames), cameras)
                        self.assertEqual(row.state, row.action)
                        self.assertEqual(row.sender, 'profile-test-human')
                        self.assertGreater(row.action_timestamp_us, row.timestamp_us)
                        tick = next(iter(row.state.values()))
                        for frame in row.frames.values():
                            self.assertAlmostEqual(tick_from_frame(frame), tick, delta=.5)
                    print(f'PASS: {profile}: {len(rows)} aligned rows, {fields} fields, {cameras} cameras, both timelines')


if __name__ == '__main__':
    unittest.main()
