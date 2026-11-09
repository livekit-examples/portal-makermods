# Working on this tutorial

This repo teaches the full HITL loop with Portal: collect demonstrations,
run Dream-1 or train ACT, hand off between policy and human, and record corrections for retraining.
Keep it small and readable. The README leads with the numbered `tutorial/`.

## Structure

- `src/robot.py`, `src/teleoperator.py`, and `src/recorder.py` show the Portal flows.
- `src/robot_rebot.py` (`uv run --group rebot robot-rebot`) reuses the robot loop with one Seeed B601-DM
  and overhead/arm cameras; `src/utils/rebot.py` uses MotorBridge's Damiao driver, loaded lazily.
- `src/utils/common.py` holds shared environment, token, and scheduling helpers.
- `src/utils/metal.py` drives the Metal pair through Dream Labs' `robots` package (imported lazily):
  followers energized, leaders read-only; `src/utils/` also holds preview, hotkey, and task-prompt details.
- `src/policies/act/` contains training (`train.py`), the operator (`run.py`, `uv run act`), and a SkyPilot example.
- `src/policies/dream_1/` runs Dream Labs' Dream-1 (`run.py`, `uv run dream-1`) with the dream client's `metal` IK.
- `src/policies/dream_1_tasks/` drives the single ReBot with a switchable task (`uv run dream-1-tasks`); its
  RebotMapping uses a custom Dream arm file and pads the base model's absent left arm/view, never Portal data.
  It answers the `dream.*`
  RPCs named in `src/utils/common.py`: set the task, pause, resume, hand control to the teleoperator or the webcam
  hand teleop (`src/hand_teleop.py`, identity from `hand_teleop_identity()`), take it from the hand teleop.
- `src/voice_agent.py` (`uv run --group voice voice-agent`) is a LiveKit Agents voice agent on agent-starter-python's
  pipeline that calls those RPCs, plus the robot's `home` RPC and its `park` RPC with payload `"park"`, which only
  parks (never powers up). It is not a Portal peer and never sends actions.
- `src/task_monitor.py` (`uv run --group monitor task-monitor`) is a Portal observer that scores dream-1-tasks' runs
  with LeRobot's TOPReward (Qwen3-VL), clears the control gate when a run is done or stuck, and reports to the voice
  agent over `monitor.report`. Its model loads only inside `Scorer`, so the rest imports without the monitor group.
- `test/` checks behavior without hardware.

Run the three peers as `uv run robot`, `uv run teleoperator`, and `uv run recorder`
(entry points in `pyproject.toml`); policies run as scripts under `src/policies/`.
Keep explanations in the tutorials rather than creating a separate docs layer.
Update affected chapters with behavior changes.

## Rules

- All peers load `portal.yaml` through `utils.common.portal_config`; do not duplicate schema field lists in Python.
  Top-level fields are Metal; `profiles.rebot` overrides them for ReBot. Recorder, monitor, and ACT use
  `PORTAL_PROFILE`; the hardware-specific apps select their profile explicitly.
- Model tensors use training dataset order, read from `meta/info.json`.
- Control is explicit: `c` selects an operator, `p` parks or powers up the arms. No background auto-claim loop.
  A task change releases Dream-1's control first; only an explicit resume (`c`, or "continue") gives it back.
  The task monitor only ever releases control (clears the gate); it never gives control to anyone.
- Keep recording separate from the control loops. Preserve action anchors and
  both archive timelines; let Portal stamp synchronized action send times.
- Run blocking hardware or inference off the event loop. Device calls must be
  serialized and finished before disconnecting hardware.
- The Metal arms are brakeless: never disable the motors on a fault. They switch off only through
  a park (`p`, or any robot exit): walk to the pose captured at startup, verify, then disable. A park
  that cannot finish leaves them holding. The robot's `home` RPC is that walk without the disable: it
  releases control, never parks or powers up. Opening is both arms or neither.
- Preserve the soft-limit clamp, the target-step limiter, and both-arms-or-nothing reads. A small clamp on
  tracking error limits torque under MIT control and can immobilize joints.
- CAN links come from `.env` with no defaults, so a missing setting fails instead of swapping arms.
- Check installed library APIs when restoring older examples.

## Checks

```bash
uv run --group train python -m unittest test.test_act
uv run python -m test.test_teleop
uv run python -m test.test_resilience
uv run python -m unittest test.test_dream1 test.test_dream1_tasks test.test_task_monitor
uv run python -m unittest test.test_rebot
uv run python -m unittest test.test_hands
# With livekit-server --dev:
uv run --group voice python -m test.test_voice_tasks
uv run --group voice python -m test.test_monitor_voice
uv run python -m test.test_pubsub
uv run python -m test.test_recorder
uv run python -m test.test_handoff
```
