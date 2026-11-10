"""Same-origin browser UI and a per-connection Python Portal Operator.

Video travels directly over LiveKit. WebSocket input becomes anchored Portal actions;
this bridge never writes MuJoCo controls. The web access code stays server-side.
"""
from __future__ import annotations

import asyncio
from collections import deque
import datetime
import json
import math
import os
import secrets
import time
from urllib.parse import urlsplit

from aiohttp import web, WSMsgType

from utils.mujoco_sim import ARMS, SETUPS
from livekit import api
from livekit.portal import Operator, OperatorConfig

from utils.common import ROOT, mint_token, portal_config, required_env, DREAM_TASKS_IDENTITY
from utils.sim_tasks import DreamTaskSync, task_text

DIST = ROOT / "web" / "sim" / "dist"


def viewer_token(identity, room):
    return (api.AccessToken(required_env("LIVEKIT_API_KEY"), required_env("LIVEKIT_API_SECRET"))
            .with_identity(identity)
            .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=False,
                                        can_publish_data=False, can_subscribe=True))
            .with_ttl(datetime.timedelta(hours=6)).to_jwt())


class BrowserServer:
    def __init__(self, room, cfg, limits, robot, inbox, access_code, *, profile="widowx"):
        self.room, self.cfg, self.limits, self.robot, self.inbox = room, cfg, limits, robot, inbox
        self.profile = profile
        self.access_code = access_code
        self.sessions = {}
        self.sockets = set()
        self.status = {}
        self.runner = None
        self.tasks = DreamTaskSync(robot, profile)
        self.task_sync = None
        self.app = web.Application(client_max_size=16 * 1024, middlewares=[self.security])
        self.app.add_routes([web.post("/api/login", self.login), web.get("/api/session", self.session),
                             web.get("/ws", self.websocket), web.get("/healthz", self.health)])
        self.app.on_shutdown.append(self.shutdown)
        if DIST.exists():
            self.app.router.add_get("/", self.index)
            self.app.router.add_static("/assets", DIST / "assets")

    @web.middleware
    async def security(self, request, handler):
        origin = request.headers.get("Origin")
        if origin and (urlsplit(origin).netloc != request.host or urlsplit(origin).scheme not in ("http", "https")):
            raise web.HTTPForbidden(text="same-origin requests only")
        if request.path in ("/ws", "/api/session"):
            expires = self.sessions.get(request.cookies.get("sim_session"), 0)
            if expires <= time.monotonic():
                raise web.HTTPUnauthorized(text="Enter the browser access code")
        response = await handler(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    async def health(self, request):
        return web.json_response({"ready": bool(self.status)}, status=200 if self.status else 503)

    async def index(self, request):
        return web.FileResponse(DIST / "index.html")

    async def login(self, request):
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("code"), str):
            raise web.HTTPBadRequest()
        if not secrets.compare_digest(payload["code"].encode(), self.access_code.encode()):
            await asyncio.sleep(.3)
            raise web.HTTPUnauthorized(text="Incorrect access code")
        now = time.monotonic()
        self.sessions = {key: expiry for key, expiry in self.sessions.items() if expiry > now}
        if len(self.sessions) >= 64:
            raise web.HTTPTooManyRequests(text="Too many sessions; restart the server to clear them")
        session = secrets.token_urlsafe(32)
        self.sessions[session] = now + 6 * 3600
        response = web.json_response({"ok": True})
        response.set_cookie("sim_session", session, httponly=True, samesite="Strict", max_age=6 * 3600,
                            secure=request.secure or request.headers.get("X-Forwarded-Proto") == "https")
        return response

    async def session(self, request):
        return web.json_response({"ok": True})

    async def websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=5, max_msg_size=16 * 1024)
        await ws.prepare(request)
        self.sockets.add(ws)
        identity = "browser-" + secrets.token_hex(6)
        op = Operator(portal_config(OperatorConfig, self.room, self.profile))
        observations = deque(maxlen=60)
        latest = None
        generation = 0
        changed_at = 0
        sender = None
        recording = False

        def observation(obs):
            nonlocal latest
            latest = (obs, time.monotonic())
            observations.append(obs.timestamp_us)

        def changed(_):
            nonlocal generation, changed_at
            generation += 1
            changed_at = op.now_us()
            observations.clear()

        op.on_observation(observation)
        op.on_active_operator_changed(changed)

        async def updates():
            while not ws.closed:
                obs = latest[0] if latest else None
                fresh = latest is not None and time.monotonic() - latest[1] < .75
                await ws.send_json({"type": "state", "state": obs.state if obs else None,
                                    "anchor": obs.timestamp_us if obs else None,
                                    "fresh": fresh, "generation": generation,
                                    "active": op.active_operator(), "operators": self.robot.operators(),
                                    "dream_task": self.tasks.snapshot(),
                                    "busy": self.inbox.busy, "recording": recording, **self.status})
                await asyncio.sleep(.1)

        try:
            await op.connect(required_env("LIVEKIT_URL"), mint_token(identity, self.room))
            await ws.send_json({"type": "ready", "identity": identity, "limits": self.limits, "setups": SETUPS,
                                "profile": self.profile, "arm_label": ARMS[self.profile],
                                "task": self.tasks.task,
                                "cameras": list(self.cfg.video_tracks),
                                "url": os.environ.get("LIVEKIT_BROWSER_URL") or required_env("LIVEKIT_URL"),
                                "token": viewer_token("view-" + identity, self.room)})
            sender = asyncio.create_task(updates())
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(message.data)
                    if not isinstance(data, dict):
                        raise ValueError("expected a JSON object")
                    kind = data.get("type")
                    if kind == "control":
                        target = data.get("identity")
                        if target is not None and target not in self.robot.operators():
                            raise ValueError("operator is no longer connected")
                        if self.inbox.busy and target is not None:
                            raise ValueError("wait for the simulation transition")
                        if target == DREAM_TASKS_IDENTITY:
                            await self.tasks.handoff(data.get("task"))
                        else:
                            await op.set_active_operator(target)
                    elif kind == "task":
                        if recording:
                            raise ValueError("end or discard this take before changing its task")
                        await self.tasks.set_task(data.get("task"))
                    elif kind == "action":
                        anchor = data.get("anchor")
                        if (op.active_operator() != identity or data.get("generation") != generation
                                or not isinstance(anchor, int) or anchor not in observations
                                or anchor <= changed_at or op.now_us() - anchor > 750_000):
                            continue
                        values = self.inbox.validate(data.get("values", {}))
                        op.send_action(values, in_reply_to_ts_us=anchor)
                    elif kind == "motion":
                        if recording:
                            raise ValueError("end or discard this take before resetting or going home")
                        await op.perform_rpc("sim.motion", data.get("command", ""))
                    elif kind == "view":
                        # A shared spectator camera, not an action or policy camera.
                        view = data.get("view", {})
                        if not isinstance(view, dict) or set(view) != {"azimuth", "elevation", "distance"}:
                            raise ValueError("invalid camera view")
                        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                               for v in view.values()):
                            raise ValueError("invalid camera view")
                        self.inbox.view = view
                    elif kind == "record":
                        mark = data.get("mark")
                        if mark not in ("recording", "idle", "discard"):
                            raise ValueError("invalid recording marker")
                        if not latest or time.monotonic() - latest[1] > .75:
                            raise ValueError("wait for fresh observations")
                        task = task_text(data.get("task"))
                        acknowledgments = await op.send_keypoint(mark, {"task_description": task},
                                                                timestamp_us=latest[0].timestamp_us)
                        if not acknowledgments:
                            raise ValueError("No recorder acknowledged the marker; start the recorder first")
                        recording = mark == "recording"
                    else:
                        raise ValueError("unknown browser message")
                except (ValueError, TypeError, KeyError) as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": f"Portal request failed: {exc}"})
        except Exception as exc:
            if not ws.closed:
                await ws.send_json({"type": "error", "message": str(exc)})
        finally:
            if sender:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            try:
                if op.active_operator() == identity:
                    await op.set_active_operator(None)
                if recording and latest:
                    await op.send_keypoint("discard", {}, timestamp_us=latest[0].timestamp_us)
            except Exception:
                pass
            try:
                await op.disconnect()
            finally:
                op.close()
                self.sockets.discard(ws)
                await ws.close()
        return ws

    async def shutdown(self, app):
        if self.task_sync:
            self.task_sync.cancel()
            await asyncio.gather(self.task_sync, return_exceptions=True)
        await asyncio.gather(*(ws.close(code=1001, message=b"Simulation stopping") for ws in list(self.sockets)))

    async def start(self, host, port):
        if not (DIST / "index.html").exists():
            raise RuntimeError("Build the browser first: cd web/sim && npm ci && npm run build")
        self.runner = web.AppRunner(self.app, shutdown_timeout=5)
        await self.runner.setup()
        await web.TCPSite(self.runner, host, port).start()
        self.task_sync = asyncio.create_task(self.tasks.run())

    async def close(self):
        if self.runner:
            await self.runner.cleanup()
