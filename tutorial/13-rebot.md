# 13. One Seeed reBot B601-DM

This rig uses one B601-DM (Damiao motors), one overhead USB camera, and one
camera on the arm. Run `robot-rebot` for the hardware and `dream-1-tasks` for
the policy. The original `robot`, `teleoperator`, `hand-teleop`, and `dream-1`
apps still describe the Metal rig; do not join them to the ReBot room.

## 1. Prepare the arm and cameras

Complete Seeed's [B601-DM assembly and zero calibration](https://wiki.seeedstudio.com/rebot_b601_dm_getting_started/)
first. Motor IDs must be 1..7, with feedback IDs 0x11..0x17. The first three
motors are 4340P; the wrist and gripper motors are 4310, following Seeed's
[DM hardware configuration](https://github.com/vectorBH6/reBotArm_control_py/blob/main/config/rebotarm_dm.yaml).
This app uses MotorBridge directly, in POS_VEL mode, without the SDK's
background control thread or automatic return-to-zero on disconnect.

Install the optional driver:

```bash
uv sync --group rebot
```

Copy `.env.example` to `.env` if needed and set the LiveKit connection as in
[chapter 1](01-first-recording.md). Use a separate room from the Metal rig:

```dotenv
LIVEKIT_ROOM=portal-rebot
PORTAL_PROFILE=rebot
REBOT_CAN=/dev/serial/by-id/YOUR_DAMIAO_ADAPTER
REBOT_TRANSPORT=dm-serial
REBOT_CAM_OVERHEAD=/dev/video0
REBOT_CAM_ARM=/dev/video2
```

The stock Damiao serial bridge runs at 921600 baud. For a configured SocketCAN
adapter, set `REBOT_TRANSPORT=socketcan` and explicitly set `REBOT_CAN` to its
link name. There is no default arm or camera device. On macOS, camera settings
can be numeric indices. `REBOT_CAM_WIDTH` and `REBOT_CAM_HEIGHT` default to
640 and 480; cameras request MJPG at the Portal frame rate (30 Hz by default).

Measure the gripper motor angle at closed and open using Seeed's calibration
tools and set `REBOT_GRIPPER_CLOSED_RAD` and `REBOT_GRIPPER_OPEN_RAD`. Both are
required radians in the configured -5.2..0 motor range, and must differ.
Neither a zero calibration nor a gripper calibration is performed at startup.
Portal carries the gripper as **0 closed to 1 open**, not millimetres.

## 2. Configure ReBot kinematics for Dream-1

The B601-DM is not a Metal arm. `dream-1-tasks` requires `REBOT_ARM_FILE`, the
absolute path of Dream's custom arm JSON, referencing the B601-DM URDF.
Get the matching URDF from Seeed's
[SDK repository](https://github.com/vectorBH6/reBotArm_control_py/tree/main/urdf),
then open Dream's tool-point setup:

```bash
uv run dreamcli tcp
```

Load the B601-DM URDF, select `base_link` as the base and `link6` as the flange,
and mark the tool center between the closed fingertips and its approach/jaw
directions. Save the arm file and use the path the tool prints:

```dotenv
REBOT_ARM_FILE=/absolute/path/to/arms/rebot.json
DREAM_API_KEY=YOUR_DREAM_KEY
```

The file describes `urdf`, `base`, `flange`, `tcp`, `approach`, `jaw`, and a
six-joint `rest` pose, all in URDF order/units. The URDF path is relative to the
JSON file. Keep the JSON and URDF together when copying them to the policy
host. Confirm the URDF's joint signs/zeros match your calibrated hardware;
`rest` is an IK preference, separate from the robot's startup parking pose.

The installed Dream client and base checkpoint use a bimanual interface.
`RebotMapping` supplies the real arm as `right_arm` and the real arm camera as
`wrist_r`. It pads the absent left arm with a fixed pose from the arm file and
a black `wrist_l` image, and discards all left-arm actions. Only seven real
state/action values and the two real videos cross Portal or reach a recorder.
This is a compatibility adapter for the base model, not a checkpoint trained
specifically on the single-arm rig; inspect its predictions before driving.

## 3. Start and check the loop

Start with the calibrated arm powered off at a supported rest pose, then apply
power and run:

```bash
uv run --group rebot robot-rebot
```

The app reads every motor before enabling, seeds the position targets from
that pose, and captures it as home. An already enabled arm is rejected; after
a failed park, support it and cycle power before restarting. It begins with
no active operator. In a second terminal:

```bash
uv run dream-1-tasks --dry-run --task "put the blue cube in the box"
```

Press `c` to inspect the seven proposed targets. Both camera tracks and a
complete state must arrive before predictions are played. Stop the dry run,
then restart without `--dry-run` and press `c` when ready. `1`–`9` select
repeated `--task` presets; changing a task releases control and needs another
`c` or an explicit voice resume. The [voice agent](10-voice-tasks.md) can set
tasks, pause, resume, home, and park this robot using the existing RPCs.
Human handoff RPCs remain available for a ReBot-compatible operator, but the
repository's Metal leader and webcam-hand operators are not ReBot drivers.

`p` releases control and walks back to the startup pose before disabling
motors. `p` again powers up, holding the current resting pose. `home` does the
same walk while keeping motors on. Exiting the robot also parks first. Failed
reads suppress actions; a failed or blocked park leaves motors holding rather
than disabling them. All device calls, including connection and cleanup, run
serially on the hardware worker.

The robot clamps to ReBot's joint limits and ramps target changes by at most
`REBOT_MAX_STEP_RAD` (default 0.03) per command. Motor velocity is limited by
`REBOT_SPEED_RAD_S` (default 0.5); home/park targets move at
`REBOT_PARK_DEG_S` (default 10 degrees/s). These settings must be positive.

For observers, set `PORTAL_PROFILE=rebot` on their hosts too:

```bash
uv run recorder
uv run --group monitor task-monitor --dry-run
```

The recorder still needs a compatible operator to start/end takes; starting
it alone does not record an episode. The task monitor continues to watch the
overhead camera and only releases control. ACT must use a dataset trained on
this seven-value/two-camera contract; Metal checkpoints are incompatible.

## 4. Hardware-free checks

```bash
uv run python -m unittest test.test_rebot test.test_dream1 test.test_dream1_tasks
uv run python -m test.test_resilience
```

These check driver calls, missing feedback, limits, parking, profile selection,
and model mapping. They do not establish physical calibration, camera
placement, or task success on the real arm.

If state arrives but synchronized observations do not, compare the transport
test with `PORTAL_TEST_CODEC=mjpeg`. To try MJPEG on the real rig, replace both
video entries under `profiles.rebot` with `{name: overhead_camera, codec: mjpeg}`
and `{name: arm_camera, codec: mjpeg}` on every host; MJPEG entries must omit the
H264-only bitrate and screencast settings. This helps separate media decoding
or timestamp problems from the arm/schema path.

---

[Tutorial index](../README.md#the-tutorial)
