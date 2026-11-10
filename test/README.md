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
| `test_hands_sim` | WidowX/YAM synthetic hand gestures, tool-relative roll/depth on both arms, normalized grip, native joint order and TCP, speed/floor limits, hold/reacquire, schema rejection, and simulation-home versus hardware-park RPCs (requires sim group) |
| `test_hands` | Webcam hand teleop without a camera: the grip gesture, hand pose, gripper POV at rotated hand/tool poses, fixed base mapping, the IK's floor and place-before-turn, and glitches letting go |
| `test_pubsub` | Discovery, control gate, observation transport |
| `test_recorder` | Episode markers and frame/state/action alignment after an archive round trip |
| `test_recorder_profiles` | Explicit recorder profile overrides, ReBot compatibility, and single/dual-arm viewer layouts |
| `test_recorder_profiles_live` | Actual recorder archive round trips for dual WidowX AI and single ReBot, including all cameras and both timelines |
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

## MuJoCo WidowX / YAM simulation

`uv run --group sim --inexact python -m unittest test.test_mujoco_sim test.test_yam_sim test.test_workbench test.test_robocasa_objects test.test_sim_process test.test_sim_web test.test_hands_sim`
checks real model dynamics, limits, reset, handoff barriers, authentication, and
viewer-token permissions without a display or server. Workbench checks cover
desk-item settling/reset with both arm pairs and overhead-camera framing. RoboCasa checks verify
source hashes, visual/contact separation, prop mass, settling/reset with both arm
pairs, and a test probe falling into the hollow mug, bowl and tray.

`uv run --group sim --inexact python -m test.test_mujoco_render` checks four real
RGB renders with both renderers for each arm pair, including process startup,
shutdown, orbit and frame-buffer ownership. Use a macOS graphical session; Linux
NVIDIA needs Vulkan for Filament and `MUJOCO_GL=egl` for classic. YAM tests also verify IK/TCP/jaw, units and independent motion.

For live browser-bridge movement, policy takeover, timeout, and reset checks, see
[chapter 14](../tutorial/14-mujoco-sim.md) and `test.test_sim_handoff`. It requires
a disposable running simulator and demo policy, and LiveKit frame metadata
support (tested with 1.13.11; 1.9.3 drops the required camera timestamps).

WidowX AI / YAM Dream mapping and browser tasks (no API requests):

```bash
uv run --group sim --inexact python -m unittest test.test_dream1_widowx test.test_sim_tasks
# With a disposable livekit-server --dev:
SIM_TASK_TEST_LIVEKIT=ws://127.0.0.1:7880 uv run --inexact python -m unittest test.test_sim_task_rpc
```

These check joint/gripper mapping, MuJoCo-vs-URDF FK, task delivery on late
connection/reconnection, explicit resume, and pausing before task changes.

Recorder profile checks:

```bash
uv run --inexact python -m unittest test.test_recorder_profiles
# With a disposable recent livekit-server --dev:
RECORDER_TEST_LIVEKIT=ws://127.0.0.1:7880 uv run --inexact python -m unittest test.test_recorder_profiles_live
```
