# Hardware-free checks

Run from the repo root. ACT checks need the training dependency group:

```bash
uv run --group train python -m unittest test.test_act
uv run python -m test.test_teleop
uv run python -m test.test_resilience
uv run python -m unittest test.test_dream1 test.test_dream1_tasks test.test_task_monitor
uv run python -m unittest test.test_rebot
uv run python -m unittest test.test_hands
```

For transport and archive checks, start `livekit-server --dev` in another
terminal. Test credentials default to that local server.

```bash
uv run python -m test.test_pubsub
uv run python -m test.test_recorder
uv run python -m test.test_handoff
uv run python -m test.test_degraded
uv run --group voice python -m test.test_voice_tasks
uv run --group voice python -m test.test_monitor_voice
```

| Suite | Checks |
|---|---|
| `test_act` | Dataset field order, tensor mapping, training arguments, existing-output handling |
| `test_teleop` | Control cycling, frame anchors, task selection, keyboard input, preview |
| `test_resilience` | Per-arm CAN failure recovery, both-arms-or-nothing state, target-step and soft limits, park and home requests |
| `test_dream1` | Dream-1 embodiment, camera roles, frame encoding, unit mapping, chunk layout |
| `test_rebot` | Single-arm/two-camera profile, ReBot units and limits, MotorBridge calls, read failures, park verification, Dream padding and right-only actions |
| `test_dream1_tasks` | A task change pauses first and never starts motion; pause, resume, teleop and hand-teleop handoff, hand teleop off; task sources |
| `test_hands` | Webcam hand teleop without a camera: the grip gesture, the hand's position and turn (wherever it sits in the image), the mapping onto each arm, the IK's floor and place-before-turn, and glitches letting go |
| `test_pubsub` | Discovery, control gate, observation transport |
| `test_recorder` | Episode markers and frame/state/action alignment after an archive round trip |
| `test_handoff` | Human → policy → human actions recorded in one episode |
| `test_degraded` | Alignment under jitter, drops, and stalls |
| `test_voice_tasks` | The voice agent's RPCs to a real operator: task, pause, resume, teleop and hand-teleop handoff, hand teleop off, home, park, stop words |
| `test_task_monitor` | The monitor's clip, the wait before done, stuck verdicts, release of control, dry run, handover; a stub model |
| `test_monitor_voice` | The monitor over a real server: a run done, one stuck, one handed over; the voice agent's reports and announcements, and the newest report in every LLM call |

Synthetic peers encode tick indices in joint state, a white bar in video, and
returned actions. The archive reader checks they agree using exact timestamp
selection (`max_stale_us=0`). Normal conversion permits 200 ms of staleness.

Portal 0.3.0rc1 has a known timestamp regression after closely spaced ticks in
the `stalls` profile: a frame can carry its neighbour’s timestamp. That profile
may fail even when the other suites pass. A mismatched row is a failure, not an
acceptable effect of network degradation.

`PORTAL_TEST_PROFILE` selects one degradation profile; `PORTAL_TEST_CODEC`
overrides the video codec; `PORTAL_TEST_KEEP_RRD` retains test archives.
`netsim.py` provides optional macOS pf/dummynet transport shaping. Inspect
`uv run python -m test.netsim show wan` before applying it: it modifies the
system packet filter and requires sudo for changes.
