"""Recorder side: a Portal observer that writes the session's `.rrd`.

Run with `uv run recorder`. It subscribes to state, camera frames, the actions
of whichever operator drives, and every keypoint, and writes them with Portal's
`RrdSink`. There is nothing to press and no window to look at. An observer
cannot drive, so this process cannot move the arm however it is started.

`uv run recorder --observe-only` writes nothing and shows the same streams live
in a Rerun window instead: a quick look at what is in the room.

The archive layout is Portal's: see its docs/03-data-recording.md.
"""
from __future__ import annotations

import argparse
import asyncio

from utils.common import portal_config, RECORDER_IDENTITY, env_int, env_str, load_env, mint_token, required_env
from utils.rerun_view import OBSERVER_VIEWER_PORT
from livekit.portal import Observer, ObserverConfig
from livekit.portal.recording import RrdSink
import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def layout(session) -> rrb.Blueprint:
    """Cameras across the top; under them each arm's joints and gripper, then the actions; Portal's metrics last."""
    def path(name: str) -> str:
        return f"/observation/{rr.escape_entity_path_part(name)}"

    def plot(title: str, names: list[str]) -> rrb.TimeSeriesView:
        return rrb.TimeSeriesView(name=title, origin="/observation", contents=[f"+ {path(n)}" for n in names])

    fields = [f.name for f in session.state_schema]
    arms = []
    for side in ("left", "right"):
        mine = [n for n in fields if n.startswith(f"{side}_")]
        arms.append(rrb.Vertical(
            plot(f"{side} joints (rad)", [n for n in mine if "gripper" not in n]),
            plot(f"{side} gripper (mm)", [n for n in mine if "gripper" in n]),
            row_shares=[3, 1],
        ))
    cameras = [rrb.Spatial2DView(name=t.name, origin=path(t.name)) for t in session.video_tracks]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*cameras),
            rrb.Horizontal(*arms, rrb.TimeSeriesView(name="actions", origin="/action")),
            rrb.Horizontal(  # RrdSink logs these once a second
                rrb.TimeSeriesView(name="round trip (us)", origin="/metrics/rtt_us"),
                rrb.TimeSeriesView(name="clock sync (us)", origin="/metrics/time_sync"),
                rrb.TimeSeriesView(name="received (total)", origin="/metrics/transport"),
            ),
            row_shares=[3, 2, 1],
        ),
        collapse_panels=True,
    )


class LiveSink(RrdSink):
    """Portal's `.rrd` layout, sent to a spawned Rerun viewer instead of a file."""

    def open(self, session) -> None:
        self._rec = rr.RecordingStream("livekit-portal", recording_id=session.session)
        self._rec.spawn(port=OBSERVER_VIEWER_PORT, memory_limit="25%")  # the viewer drops the oldest data past this
        self._rec.send_blueprint(layout(session))
        self._now_us = session.now_us

    def write_frame(self, track: str, frame) -> None:
        """Only the newest frame per camera (static), so the viewer's memory stays flat; plots keep their history."""
        pixels = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        self._stream().log(f"observation/{rr.escape_entity_path_part(track)}",
                           rr.Image(pixels).compress(jpeg_quality=self.jpeg_quality), static=True)


def status(obs: Observer) -> str:
    m = obs.metrics()
    return (f"{m.transport.states_received} states, "
            f"{sum(m.transport.frames_received.values())} frames, "
            f"{m.transport.actions_received} actions; "
            f"{m.observer.frames_written} frames written, "
            f"{m.observer.frames_dropped} dropped by the writer")


async def main(observe_only: bool) -> None:
    load_env()

    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    obs = Observer(portal_config(ObserverConfig, room))
    obs.on_keypoint(lambda kp: print(f"[recorder] {kp.type} {kp.payload} from {kp.sender}"))

    identity = "observer" if observe_only else RECORDER_IDENTITY  # never displaces the real recorder
    print(f"[recorder] connecting to {url} as '{identity}' in room '{room}' ...")
    # connect() returns after the first sync with the robot's clock, so every row
    # is stamped on it. Start the robot first: without it, connect() raises
    # TimeSyncTimeout after 10 s.
    await obs.connect(url, mint_token(identity, room))

    if observe_only:
        sink = LiveSink("", jpeg_quality=80)
    else:
        sink = RrdSink(env_str("PORTAL_DATA_RRD_DIR", "data/sessions"),
                       jpeg_quality=env_int("PORTAL_DATA_RRD_QUALITY", 95))
    obs.record_to(sink)
    print(f"[recorder] {'showing the room in Rerun' if observe_only else f'recording to {sink.path}'}; ctrl-c to stop")

    try:
        while True:
            await asyncio.sleep(5.0)
            print(f"[recorder] {status(obs)}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[recorder] stopping ...")
    finally:
        try:
            await obs.disconnect()  # writes what is queued, then closes the file
        finally:
            obs.close()
            if sink.path:
                print(f"[recorder] wrote {sink.path}")


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--observe-only", action="store_true", help="show the room live in Rerun; write nothing")
    asyncio.run(main(parser.parse_args().observe_only))


if __name__ == "__main__":
    run()
