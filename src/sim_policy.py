"""A deliberately simple demo policy for testing Portal handoff, not a learned task policy.

It starts idle. Select policy-sim-demo in the browser to run a small waist sweep
around the observed pose; human takeover discards its trajectory immediately.
For learned inference use `PORTAL_PROFILE=widowx uv run act ...` instead.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import math
import signal
import time

from livekit.portal import Operator, OperatorConfig
from utils.common import load_env, mint_token, pace, portal_config, required_env

IDENTITY = "policy-sim-demo"


async def main(profile=None):
    load_env()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    room = required_env("LIVEKIT_ROOM")
    profile = profile or os.environ.get("SIM_ARMS", "widowx")
    if profile not in ("widowx", "yam"):
        raise ValueError("simulation profile must be widowx or yam")
    cfg = portal_config(OperatorConfig, room, profile)
    waist = "joint_0" if profile == "widowx" else "joint_1"
    op = Operator(cfg)
    latest = None
    base = None
    started = 0
    cutoff = 0

    def observed(obs):
        nonlocal latest
        latest = (obs, time.monotonic())

    def changed(identity):
        nonlocal base, cutoff
        base = None
        cutoff = op.now_us()
        print(f"[sim-policy] active operator: {identity}", flush=True)

    op.on_observation(observed)
    op.on_active_operator_changed(changed)
    try:
        await op.connect(required_env("LIVEKIT_URL"), mint_token(IDENTITY, room))
        print("[sim-policy] idle; select policy-sim-demo in the browser", flush=True)
        async for _ in pace(cfg.fps):
            if op.active_operator() != IDENTITY:
                continue
            if not latest or time.monotonic() - latest[1] > .5:
                await op.set_active_operator(None)
                continue
            obs = latest[0]
            if obs.timestamp_us <= cutoff:
                continue
            if base is None:
                base = {key: float(value) for key, value in obs.state.items()}
                started = time.monotonic()
            action = base.copy()
            offset = .18 * math.sin((time.monotonic() - started) * math.pi / 3)
            action[f"left_{waist}.pos"] += offset
            action[f"right_{waist}.pos"] -= offset
            op.send_action(action, in_reply_to_ts_us=obs.timestamp_us)
    finally:
        try:
            if op.active_operator() == IDENTITY:
                await op.set_active_operator(None)
        finally:
            try:
                await op.disconnect()
            finally:
                op.close()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("widowx", "yam"), help="defaults to SIM_ARMS or widowx")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.profile))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    run()
