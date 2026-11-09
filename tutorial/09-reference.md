# 9. Configuration and troubleshooting

Use this chapter when changing hosts, tuning settings, or diagnosing a problem.
For your first run, start with [rig setup](01-first-recording.md).

For the single B601-DM, see [ReBot setup and settings](13-rebot.md). Its
`REBOT_*` device, gripper, speed, and camera settings are separate from the
Metal settings below. Recorder, task monitor, and ACT select its wire
contract with `PORTAL_PROFILE=rebot`; `robot-rebot` and `dream-1-tasks`
select it explicitly.

## Human-only demonstrations

If you prefer to train without Dream-1, complete [chapter 1](01-first-recording.md),
then use the running teleoperator and recorder:

1. Press `c` and confirm the human is selected.
2. Reset the towel and arms, then press `r` to begin a take.
3. Demonstrate the task with the leaders.
4. Press `r` to finish. For a failed attempt, press `[` while the take is open instead.
5. Repeat steps 2–4 to collect varied successful demonstrations.
6. Press `x` in teleop, stop the recorder with Ctrl-C and wait for it to exit,
   then stop the robot with Ctrl-C; it parks the followers first.
7. Follow [chapter 2's conversion steps](02-dream-1.md#4-convert-and-inspect),
   then continue to [ACT training](03-train-act.md).

## Other host layouts

The walkthrough uses separate robot and teleop hosts. You can run all three
peers on one Linux machine if it has the required CAN adapters and cameras.
If leader and follower gripper measurements differ, give robot and teleop
separate working directories with their own `.env.local` values. Start each
process from its directory; relative output paths follow that directory.

You can also run the recorder or policy on another machine. Copy the repository,
run `uv sync`, and configure the same LiveKit server, room, and `portal.yaml`.
ACT needs its checkpoint and training dataset metadata on its inference host.

### Deploy to another host

```bash
./scripts/deploy_to_robot.sh robotuser@robot.local '~/workspace/portal-makermods'
```

Replace the host and destination path. This replaces source, `.env`, and
`.env.local` on the destination; datasets and training outputs are excluded.
Run `uv sync` there afterwards and check its hardware settings.

## CAN adapters

Dream's driver opens `/dev/...` paths as serial (SLCAN) adapters at 1 Mbps,
and any other name as a SocketCAN link, which must already be up at 1 Mbps.

**Serial adapters**

- **No unique `/dev/serial/by-id/` entries:** use `/dev/serial/by-path/` instead
  and keep each adapter in the same USB port.
- **Permission denied:** add your user to `dialout` and log in again.
- **An `slcand` service already owns the adapter:** set that service's `canN`
  link name instead of the serial path.

**candleLight (`gs_usb`) adapters**

- **In `lsusb` but no `canN`:** MakerMods Lab drives these adapters directly
  over USB and leaves them detached from the kernel driver when it exits.
  Reattach them, then bring the links up:

  ```bash
  sudo modprobe -r gs_usb && sudo modprobe gs_usb
  ```

- **`canN` names can swap** after a reboot or replug, since they follow USB
  enumeration. Pin each adapter to a name by its USB serial. List the serials:

  ```bash
  for link in /sys/class/net/can*; do echo "$(basename "$link") $(cat "$link"/device/../serial)"; done
  ```

  Then write one line per adapter to `/etc/udev/rules.d/80-metal-can.rules`
  with your serials, and replug the adapters. Each link is named and brought up
  at 1 Mbps, also after a reboot:

  ```text
  SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="SERIAL_OF_LEFT", NAME="can_left", RUN+="/usr/sbin/ip link set can_left up type can bitrate 1000000"
  SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="SERIAL_OF_RIGHT", NAME="can_right", RUN+="/usr/sbin/ip link set can_right up type can bitrate 1000000"
  ```

  Set `METAL_FOLLOWER_LEFT_CAN=can_left` and `METAL_FOLLOWER_RIGHT_CAN=can_right`,
  and confirm with `uv run robot --check-arms`.

Run one controller at a time: close MakerMods Lab, and leave Dream's
`dreamcli robot up` stopped while the Portal peers run.

## Parking and power

The robot records each follower's pose when it starts. A park walks both arms
back to that pose at `METAL_PARK_DEG_S`, checks they arrived, switches them off,
and reopens them read-only so state keeps flowing. `p` in the robot terminal,
teleop, Dream-1, or ACT parks, or powers parked arms up. Ctrl-C, or any other
exit, parks before the robot stops.

Each `p` is a `park` RPC to the robot with an empty payload, a toggle. The
payload `"park"` only parks: on parked arms it does nothing, and it cancels a
toggle still waiting, so it can never power the arms up. The voice agent's "park
the arms" sends it. Powering up stays a `p` press.

A `home` RPC to the robot (any payload) walks the arms back to that pose the
same way but does not park: they stay powered, holding there, grippers as they
were. It releases control first, so nothing drives until an operator is
selected again with `c`. On parked arms it does nothing, and it gives way to a
park or power-up already waiting, which ends at rest too. Its reply says which
happens: `"returning the arms home"`, or the park or power-up under way. The
voice agent's "home the arms" or "reset the arms" sends it.

A park or a return home stops, and the arms keep holding under power, if an arm
is unreadable, falls more than 10° behind its target (blocked), or ends more
than 3° from its rest pose. A fault never switches the motors off.

## Training alternatives

- **Hosted dataset:** add `--push` to conversion with your Hugging Face repo ID.
  For training, set `DATASET_REPO_ID` to that ID and omit `DATASET_ROOT`.
- **Cloud training:** adapt the resources and storage in
  [`skypilot.yaml`](../src/policies/act/skypilot.yaml), then launch it with
  `sky launch src/policies/act/skypilot.yaml -e DATASET_REPO_ID=you/your-dataset`.
- **Replace a converted dataset:** conversion supports `--overwrite`. Prefer a
  new repo ID for each round so earlier datasets remain available.

## Configuration

Copy `.env.example` for the settings needed to connect the rig. `src/utils/common.py`
loads repo-root `.env`, then `.env.local`, then the same files from the working
directory. `.env` fills unset values; `.env.local` overrides them, including
shell values. Relative output paths use the working directory.

| Variable | Default | Purpose |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM` | required | Same server and room for all peers |
| `PORTAL_FPS` | `portal.yaml` fps | Robot, human, and policy tick rate |
| `LOG_LEVEL` | `INFO` | Python logs |
| `RUST_LOG` | warn with noisy modules suppressed | Set in the shell before startup for Portal diagnostics |
| `PORTAL_HOTKEYS_GLOBAL` | `1` | Set `0` for terminal-only input |
| `PORTAL_DATA_TASK` | `MakerMods Metal bimanual manipulation` | Default task description |
| `PORTAL_DATA_TASKS` | unset | Semicolon-separated tasks; skips prompts, maximum nine |
| `PORTAL_DATA_RRD_DIR` | `data/sessions` | Archive output directory |
| `PORTAL_DATA_RRD_QUALITY` | `95` | JPEG quality; `0` stores decoded frames uncompressed |

### Hardware

| Variable | Default | Notes |
|---|---|---|
| `METAL_GRIPPER_CLOSED_DEG`, `METAL_GRIPPER_OPEN_DEG` | required | gripper motor angle with the jaws closed / open; per arm as `METAL_{LEFT,RIGHT}_GRIPPER_*_DEG` |
| `METAL_GRIPPER_HOLD_NM` | `0.5` | the gripper's holding torque on an object, N.m |
| `METAL_FOLLOWER_LEFT_CAN`, `METAL_FOLLOWER_RIGHT_CAN` | required on the robot host | serial path or SocketCAN link per arm |
| `METAL_LEADER_LEFT_CAN`, `METAL_LEADER_RIGHT_CAN` | required on the teleop host | serial path or SocketCAN link per arm |
| `METAL_PARK_DEG_S` | `10` | joint speed while parking or returning home |
| `METAL_CAM_OVERHEAD`, `METAL_CAM_LEFT_WRIST`, `METAL_CAM_RIGHT_WRIST` | `/dev/video0`, `/dev/video2`, `/dev/video4` | Linux camera device path. |
| `METAL_CAM_WIDTH` / `_HEIGHT` | `640` / `480` | a 16:9 mode such as 1280x720 suits Dream-1 better; it center-crops otherwise |
| `METAL_MAX_STEP_RAD` | `0.087` | maximum joint target step per command (about 5°); `0` disables the limit |

`METAL_MAX_STEP_RAD` limits how far a joint target moves per command (0.087 rad
at 30 Hz is about 2.6 rad/s); use a lower value for bring-up. Grippers are not
ramped. The joints run impedance control, so the target is ramped and the
tracking error is never clamped: a clamped error caps the torque and can stall
a joint. Every command is also clamped to the soft limits.

### Dream-1

| Variable | Default | Notes |
|---|---|---|
| `DREAM_API_KEY` | none | your Dream Labs API key; `~/.dreamlabs/credentials` also works |
| `DREAM_CHECKPOINT` | `dreamlabs/dream-1-beta` | a base model or your fine-tune |
| `DREAM_MAX_JOINT_RAD_S` | `0.5` | cap on commanded joint speed; also the ramp in after a handoff |

`uv run dream-1-tasks` reads the same settings. Its tasks come from `--task`
(repeated), then `PORTAL_DATA_TASKS`, then `PORTAL_DATA_TASK`; with none, it
starts with no task and does not move until one is set.

### Voice agent

| Variable | Default | Notes |
|---|---|---|
| `LIVEKIT_INFERENCE_API_KEY`, `LIVEKIT_INFERENCE_API_SECRET` | `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | a LiveKit Cloud project's key for Inference, when `LIVEKIT_URL` is self-hosted |
| `VOICE_STT` | `assemblyai/universal-3-6-pro` | speech-to-text on [LiveKit Inference](https://docs.livekit.io/agents/models/stt/) |
| `VOICE_LLM` | `google/gemma-4-31b-it` | the [LLM](https://docs.livekit.io/agents/models/llm/) that picks the RPC |
| `VOICE_TTS`, `VOICE_TTS_VOICE` | `fishaudio/s2.1-pro`, the starter's voice | [text-to-speech](https://docs.livekit.io/agents/models/tts/) model and voice |

The defaults are [agent-starter-python](https://github.com/livekit-examples/agent-starter-python)'s.

### Task monitor

| Variable | Default | Notes |
|---|---|---|
| `MONITOR_VLM` | `Qwen/Qwen3-VL-8B-Instruct` | the Qwen3-VL model TOPReward asks; `Qwen/Qwen3-VL-4B-Instruct` or `-2B-` on smaller GPUs |
| `MONITOR_DEVICE` | CUDA, else Apple's MPS, else the CPU | where the model runs |
| `MONITOR_CAMERA` | `overhead_camera` | the `portal.yaml` video it watches |
| `MONITOR_SUCCESS` | `0.85` | done: every score at or above this ... |
| `MONITOR_SUCCESS_HOLD_S` | `10` | ... for this many seconds; Dream-1 keeps driving meanwhile |
| `MONITOR_STUCK_S`, `MONITOR_GAIN` | `45`, `0.05` | stuck: the estimate has not risen by the gain in this many seconds |
| `MONITOR_STILL_S` | `15` | stuck: the arms have not moved in this many seconds |
| `MONITOR_EVERY_S` | `2` | seconds between scores, when the model keeps up |

Calibrate `MONITOR_SUCCESS` with `uv run --group monitor task-monitor --dry-run`; see
[chapter 12](12-task-monitor.md#3-calibrate-with-a-dry-run).

### Training

| Training variable | Default | Purpose |
|---|---|---|
| `DATASET_REPO_ID` | `local/portal-makermods` | Local or Hugging Face dataset ID |
| `DATASET_ROOT` | unset | Local dataset directory; otherwise load by repo ID |
| `OUTPUT_DIR` | `outputs/<generated run name>` | Checkpoints and training output |
| `RUN_NAME` | `act-metal-pair` | Prefix for the generated name, followed by steps, batch, LR, and chunk size |
| `NUM_STEPS` | `100000` | Training steps |
| `BATCH_SIZE` | `64` | Training batch size |
| `LR` | `5e-5` | ACT optimizer learning rate |
| `CHUNK_SIZE` | `50` | Predicted and executed action horizon |
| `DEVICE` | `cuda` | Training device |
| `NUM_WORKERS` | `16` | Dataset workers |
| `VIDEO_BACKEND` | `pyav` | Dataset video decoder; use `torchcodec` with compatible Torch/FFmpeg libraries |

Inference requires `--checkpoint` and `--dataset-root`. `--task` defaults to
`PORTAL_DATA_TASK`, then `MakerMods Metal bimanual manipulation`.
It reads LiveKit settings and `PORTAL_FPS` as the teleoperator does, selects
CUDA/MPS/CPU automatically, and waits for the `c` handoff before producing actions.

## Troubleshooting

- **The robot refuses to start: `still powered from an earlier session`.** The
  last robot crashed or its park could not finish, so the joints still hold.
  Support the arms, switch their power off and on, then start the robot.
- **An arm does not open.** Check its `METAL_*_CAN` value, the adapter's link or
  serial permissions, and the arm's 24 V power; for candleLight see
  [CAN adapters](#can-adapters). `set METAL_GRIPPER_..._DEG` means a gripper
  angle is missing from `.env`.
- **The wrong arm moves.** Left and right are swapped in `.env`: check with
  `uv run robot --check-arms` and swap the two values.
- **`the gripper stops opening at N deg`.** The open angle is past the real
  stop; set that arm's `METAL_*_GRIPPER_OPEN_DEG` to N.
- **A park stopped.** The robot names the arm and the reason; the arms hold.
  Clear the obstruction and press `p` again.
- **No motion.** Press `c` and check the active operator. All hosts must use the
  same server and room.
- **No hotkeys.** Focus the terminal of the process. Global keys need X11 on
  Linux or Accessibility permission on macOS.
- **Dream-1 refuses or stalls.** `DreamAuthError`: check `DREAM_API_KEY`.
  `pool_warming`: the model is starting. It sends nothing until all three
  cameras publish.
- **The voice agent cannot reach Dream-1 or its models.** See
  [chapter 10](10-voice-tasks.md#troubleshooting).
- **The task monitor judges too early, too late, or not at all.** See
  [chapter 12](12-task-monitor.md#troubleshooting).
- **ACT sees no observations.** Every camera the checkpoint uses must publish.
- **Low frame rate.** Compare device and tick times in the robot's 5 s line, and
  lower the rate or camera resolution if the hardware cannot keep up.
- **The recorder drops frames.** Lower the JPEG quality or resolution; check disk space.
- **The recorder or observer will not join.** Start the robot first: they wait
  for its clock.

---

Previous: [8. Handoff](08-handoff.md) · [Tutorial index](../README.md#the-tutorial) · Next: [10. Steer Dream-1 by voice](10-voice-tasks.md)
