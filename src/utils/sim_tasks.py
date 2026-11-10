"""Keep the browser's instruction on Dream-1, without ever auto-claiming control."""
import asyncio
import json

from utils.common import (DREAM_TASKS_IDENTITY, DREAM_TASK_MAX_CHARS, DREAM_STATUS_RPC,
                          DREAM_SET_TASK_RPC, DREAM_RESUME_RPC)


def task_text(value):
    if not isinstance(value, str):
        raise ValueError("task must be text")
    value = " ".join(value.split())
    if not value or len(value) > DREAM_TASK_MAX_CHARS:
        raise ValueError(f"enter one task of 1–{DREAM_TASK_MAX_CHARS} characters")
    return value


class DreamTaskSync:
    def __init__(self, robot, profile="widowx"):
        self.robot = robot
        self.profile = profile
        self.task = None
        self.applied = False
        self.message = "Waiting for a task"
        self.lock = asyncio.Lock()
        self.connection = 0

    def connection_changed(self, identity):
        if identity == DREAM_TASKS_IDENTITY:
            self.connection += 1
            self.applied = False

    def snapshot(self):
        return {"task": self.task, "applied": self.applied, "message": self.message}

    def _set(self, task):
        task = task_text(task)
        if task != self.task:
            self.task, self.applied = task, False
            self.message = "Waiting to set Dream-1 task"

    async def set_task(self, task):
        async with self.lock:
            self._set(task)

    async def rpc(self, method, **payload):
        return json.loads(await self.robot.perform_rpc(method, json.dumps(payload),
                          destination=DREAM_TASKS_IDENTITY, response_timeout_ms=3000))

    async def _sync(self):
        connection = self.connection
        status = await self.rpc(DREAM_STATUS_RPC)
        if status.get("profile") != self.profile:
            raise ValueError(f"Restart Dream-1 with --profile {self.profile} (updated dream-1-tasks required)")
        if status.get("task") != self.task:
            status = await self.rpc(DREAM_SET_TASK_RPC, task=self.task)
        if status.get("task") != self.task:
            raise ValueError("Dream-1 did not confirm the task")
        if connection != self.connection:
            raise ValueError("Dream-1 connection changed; retrying")
        self.applied = True
        self.message = "Task set on Dream-1 · Hand off to run"

    async def tick(self):
        async with self.lock:
            if DREAM_TASKS_IDENTITY not in self.robot.operators():
                self.applied = False
                self.message = "Task queued · Waiting for Dream-1" if self.task else "Waiting for a task"
            elif self.task and not self.applied:
                try:
                    await self._sync()
                except Exception as exc:
                    self.message = f"Dream-1 task pending: {exc}"

    async def handoff(self, task):
        # One serialized transaction: confirm this instruction, then explicitly resume.
        # Other browser edits wait here and subsequently pause Dream through set_task.
        async with self.lock:
            self._set(task)
            if DREAM_TASKS_IDENTITY not in self.robot.operators():
                raise ValueError("Dream-1 is not connected")
            self.applied = False
            await self._sync()
            await self.rpc(DREAM_RESUME_RPC)

    async def run(self):
        while True:
            await self.tick()
            await asyncio.sleep(.5)
