"""MuJoCo WidowX or YAM pair as a Portal Robot, plus the browser operator bridge.

Run `uv run --group sim robot-sim`; see tutorial/14-mujoco-sim.md.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import signal
import time

from livekit.portal import Robot, RobotConfig

from utils.common import (HardwareThread, ROBOT_IDENTITY, load_env, mint_token, pace,
                          portal_config, required_env)
from utils.mujoco_sim import ARMS, SETUPS, Simulation
from utils.sim_render import RENDERERS
from utils.sim_process import SimulationProcess


class ControlInbox:
    """Latest command only; transitions invalidate earlier commands and observation anchors."""
    def __init__(self, robot, validate, timeout=.75):
        self.robot, self.validate, self.timeout = robot, validate, timeout
        self.action = None
        self.received = None
        self.cutoff = 0
        self.hold = True
        self.reset = False
        self.home = False
        self.view = None
        self.setup = None
        self.busy = False

    def invalidate(self):
        self.action = self.received = None
        self.cutoff = self.robot.now_us()
        self.hold = True

    def accept(self, action):
        if (self.busy or action.sender != self.robot.active_operator()
                or action.timestamp_us <= self.cutoff
                or action.in_reply_to_ts_us is None or action.in_reply_to_ts_us <= self.cutoff
                or self.robot.now_us() - action.in_reply_to_ts_us > 750_000
                or action.in_reply_to_ts_us > self.robot.now_us()):
            return
        try:
            self.action = self.validate(action.values)
            self.received = time.monotonic()
        except ValueError:
            logging.warning("Rejected invalid simulation action from %s", action.sender)

    def expired(self):
        return self.received is not None and time.monotonic() - self.received > self.timeout

    def take(self):
        result = dict(action=self.action, hold=self.hold, reset=self.reset, home=self.home, view=self.view, setup=self.setup)
        self.action = None
        self.hold = self.reset = self.home = False
        self.view = self.setup = None
        return result


async def serve(args):
    from utils.sim_web import BrowserServer

    load_env()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    room = required_env("LIVEKIT_ROOM")
    arms = args.arms or os.environ.get("SIM_ARMS", "widowx")
    if arms not in ARMS:
        raise ValueError("SIM_ARMS must be widowx or yam")
    renderer = args.renderer or os.environ.get("SIM_RENDERER", "filament")
    if renderer not in RENDERERS:
        raise ValueError("SIM_RENDERER must be filament or classic")
    cfg = portal_config(RobotConfig, room, arms)
    worker = HardwareThread("mujoco")
    robot = Robot(cfg)
    sim = None
    browser = None
    tasks = set()
    try:
        simulation = SimulationProcess if renderer == "filament" else Simulation
        sim = await worker.run(lambda: simulation(cfg, width=args.width, height=args.height,
                                                 physics_hz=args.physics_hz, arms=arms, renderer=renderer))
        inbox = ControlInbox(robot, sim.validate_action)
        robot.on_action(inbox.accept)
        robot.on_active_operator_changed(lambda _: inbox.invalidate())

        async def release():
            inbox.invalidate()
            await robot.set_active_operator(None)

        def operator_left(identity):
            if browser:
                browser.tasks.connection_changed(identity)
            if identity == robot.active_operator():
                task = asyncio.create_task(release())
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        robot.on_operator_left(operator_left)
        robot.on_operator_joined(lambda identity: browser and browser.tasks.connection_changed(identity))

        async def request_motion(data):
            # The entire transition completes in the simulation loop, with the gate closed.
            command, _, setup = data.payload.partition(":")
            if command not in ("reset", "home") or (setup and (command != "reset" or setup not in SETUPS)):
                raise ValueError("expected home, reset, or reset:<scene setup>")
            if inbox.busy:
                raise ValueError("simulation transition already pending")
            inbox.busy = True
            try:
                await release()
                inbox.reset = command == "reset"
                inbox.setup = setup or None
                inbox.home = command == "home"
            except BaseException:
                inbox.busy = False
                raise
            return "queued; control released"

        robot.register_rpc_method("sim.motion", request_motion)
        await robot.connect(required_env("LIVEKIT_URL"), mint_token(ROBOT_IDENTITY, room))
        inbox.invalidate()
        token = os.environ.get("SIM_WEB_TOKEN")
        if not token:
            if args.host not in ("127.0.0.1", "localhost", "::1"):
                raise ValueError("set SIM_WEB_TOKEN before binding the browser server to a public interface")
            token = secrets.token_urlsafe(18)
            print(f"[sim] browser access code: {token}", flush=True)
        browser = BrowserServer(room, cfg, sim.limits, robot, inbox, token, profile=arms)
        await browser.start(args.host, args.port)
        print(f"[sim] http://{args.host}:{args.port} — MuJoCo {sim.version}, "
              f"{renderer} renderer, two {ARMS[arms]} arms; no operator selected", flush=True)
        previous = time.monotonic()
        async for _ in pace(cfg.fps):
            if inbox.expired():
                await release()
            # RPC awaits or rendering may span a handoff. The worker receives a snapshot only.
            command = inbox.take()
            transition = command["reset"] or command["home"]
            state, frames, status = await worker.run(lambda: sim.tick(**command))
            if transition:
                # A claim made during reset must not restart motion or replay old observations.
                await release()
                inbox.busy = False
            timestamp = robot.now_us()
            robot.send_state(state, timestamp_us=timestamp)
            for name, frame in frames.items():
                robot.send_video_frame(name, frame, timestamp_us=timestamp)
            now = time.monotonic()
            browser.status = {**status, "realtime_factor": (1 / cfg.fps) / max(now - previous, 1e-6)}
            previous = now
    finally:
        if browser:
            await browser.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if sim:
            try:
                await robot.set_active_operator(None)
            except Exception:
                pass
        # Complete device work before destroying its renderer/context.
        try:
            await robot.disconnect()
        finally:
            robot.close()
            if sim:
                await worker.run(sim.close)
            worker.close()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", choices=tuple(ARMS), help="arm pair; defaults to SIM_ARMS or widowx")
    parser.add_argument("--renderer", choices=RENDERERS, help="camera renderer; defaults to SIM_RENDERER or filament")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--physics-hz", type=int, default=240)
    args = parser.parse_args()
    try:
        asyncio.run(serve(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    run()
