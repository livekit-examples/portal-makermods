# 11. Teleop with your hands and a webcam

`src/hand_teleop.py` is an operator like the leader-arm teleop, driven by a laptop webcam instead.
MediaPipe tracks up to two hands; your right hand drives the right arm, your left hand the left.
`PORTAL_PROFILE` (or `--profile`) selects the rig: `metal` for the pair, `rebot` for
one B601-DM driven by your right hand, or `widowx` / `yam` for the simulated pairs.
ReBot ignores the left hand. The explicit flag overrides `PORTAL_PROFILE`, including `.env.local`.

- **Grip:** curl middle, ring and pinky all the way into your palm, leaving thumb and index as a
  gripper. After three frames the arm follows that hand. A half curl does nothing.
- **Gripper:** the gap between thumb and index tip sets it: 3 cm or less closed, 9 cm or more open.
- **Let go:** lift any of the three fingers out of your palm. Nothing more is sent for that arm, and
  the robot holds it.
- **Motion:** the arm follows how far your hand has moved since you gripped, never where your hand
  is. Movements use the **gripper's point of view**: closer to the webcam advances along its fingers;
  farther away retracts. Left/right translation is reversed: your right moves toward tool -X,
  and your left toward tool +X. Vertical translation is also reversed: up moves toward tool +Y,
  and down toward tool -Y. Turning the
  gripper changes the direction of subsequent moves. To move farther than your reach, let go,
  move back, and grip again, as with a mouse.
- **Fine moves:** also as with a mouse, slow moves count for less: under about 4 cm/s the arm moves
  0.4 times as far as your hand, above 20 cm/s as far. Near an object, go slowly and the arm moves in
  small steps; to cross the workspace, move briskly.
- **Holding still:** a still hand holds the arm still. The target moves only once your hand has
  taken it 1 mm (or turned it 1.5°), so the webcam's jitter does not reach the arm.
- **Turn:** twist your wrist around the wrist-to-knuckles axis and the gripper rolls around its
  pointing axis, regardless of whether your hand starts upright, sideways, or tilted. Clockwise
  looking along your hand toward the fingers is clockwise looking out along the gripper. Tilts map
  to the gripper's other local axes, about its own tip. The first 5° do nothing, so a hand that only moves keeps
  the gripper steady. A turn the arm cannot make where it is stops the gripper there instead of
  moving the arm, and bringing your hand back brings the gripper back. Past 90° from the grip,
  turning back acts at once; to turn farther, let go, turn back, and grip again.

Picture yourself looking out along the controlled gripper's fingers. Forward may point down at
the table or across it; it follows the gripper, independently for each arm. Translation uses the
latest joint-limited command's orientation. IK keeps the tip in place while rolling, subject to
reach and joint limits; this requests a tool rotation rather than directly assigning a wrist joint.
Depth comes from how large your palm looks, so forward and back are noisier than the other
directions and smoothed more.

Use `--frame base` (or `HAND_FRAME=base`) for the original controls: toward the camera moves along
the robot base's forward axis, and rotations follow camera axes. `--frame tool` is the default
for all four profiles. The flag overrides `HAND_FRAME`, including `.env.local`.

## 1. Try it at your desk

```bash
uv run --group hands hand-teleop --offline
```

No robot or LiveKit: both arms start at the IK's rest pose and nothing is sent. The first run
downloads MediaPipe's hand model to `~/.cache/portal-makermods/` and compiles the IK (several
seconds; cached after). On macOS, allow camera access for the terminal.

The rerun window mirrors your webcam. A hand's skeleton is yellow while seen, green while it holds
its arm, and the label above it shows the selected frame, the target's displacement in **base XYZ
millimetres**, its turn in degrees, and its gripper. These coordinates stay in the base frame even
when the controls follow the tool. Roll it and `turn` grows; slide it toward the edge of
the image without turning it and `turn` should stay near 0 (if it climbs, set `HAND_CAMERA_HFOV_DEG`
to your webcam's field of view). Until a hand grips, its label shows the three fingers' curl; all three
must drop below 1.0, which takes the fingertips folded into the palm.

## 2. Drive the robot

With the robot running (chapter 1), on any machine with a webcam and the repo's `.env`:

```bash
uv run --group hands hand-teleop
```

For the single ReBot, use the same LiveKit URL and room as `robot-rebot`, and run:

```bash
uv run --group hands hand-teleop --profile rebot
```

This selects `overhead_camera` and `arm_camera`, Dream's built-in B601-DM IK,
and the robot's normalized gripper units. It also works with `--offline`.
The startup line prints the selected profile and expected camera tracks. A
Metal profile cannot synchronize the ReBot's observations because it expects
a second arm and different wrist tracks.

For either simulated pair, start `robot-sim` as in [chapter 14](14-mujoco-sim.md),
then run hand teleop on your webcam machine in the **same LiveKit room**:

```bash
uv run --group hands hand-teleop --profile widowx
# Or, for robot-sim --arms yam:
uv run --group hands hand-teleop --profile yam
```

Both profiles also support `--offline`. They load the same local custom-arm
URDF/TCP configuration as Dream's simulation mapping, with each arm's native
joint order and limits. Both grippers use **0 closed .. 1 open** on Portal;
the hand follower's internal 0..100 span is converted at the boundary.
The preview includes overhead, both wrist cameras, the scene camera, and your
webcam. The simulator may run on a cloud GPU while hand teleop stays local.
`SIM_ARMS` does not select the hand operator's profile; use `--profile` or
`PORTAL_PROFILE` to match it explicitly.

For simulation, **p returns the arms home and releases control** using `sim.motion`;
press `c` again to resume hand control. It does not reset the objects. For Metal
and ReBot, `p` keeps the existing hardware park/power behavior.

Its window shows the robot's cameras beside the webcam, apart from any window the teleoperator or
`recorder --observe-only` has open. Press `c` to take control (and again to release it), as in a
policy; `p` parks hardware or returns the simulation home, `x` quits. With [chapter 10](10-voice-tasks.md)'s voice agent and
`dream-1-tasks` running, saying "hand teleop" hands it control too, and "back to Dream-1" takes it back.
Gripping does nothing until this operator has control and a fresh robot view. Each time control
comes back, the next grip starts from where the arms are.

Safety, on top of the robot's own step limiter and soft limits:

- The target moves at most `HAND_MAX_SPEED_M_S` and the joints `HAND_MAX_JOINT_RAD_S`.
- A floor at `HAND_TABLE_Z_M` (each arm's base frame; 0 is the underside of its base) keeps the
  tool point and the whole arm above the table. Pushing into it slides along it.
- An arm lets go when its hand jumps 8 cm or turns 45° in one frame (a glitch, or two hands
  swapped; the turn only counts while the gripper turns with the hand), when the webcam stalls, or
  when the robot view goes stale.
- There is no arm-to-arm guard: keep the arms' paths apart yourself.

The gripper takes over softly: on a new grip it holds until your pinch reaches its opening, so
gripping again with an open hand does not drop what the arm holds. The label says so while it waits.

## 3. Settings

| Setting | Default | Purpose |
|---|---|---|
| `PORTAL_PROFILE` / `--profile` | `metal` | Rig: `metal`, `rebot`, `widowx`, or `yam`; selects cameras, joints, units, and IK |
| `HAND_CAMERA` | `0` | Webcam index (`/dev/videoN` on Linux), or a video file played in real time |
| `HAND_FRAME` / `--frame` | `tool` | `tool` maps motion to the gripper's POV and wrist twist to local roll; `base` restores fixed robot-base directions |
| `HAND_ROTATE` / `--rotate` | `full` | `full` turns the gripper as the hand turns; `yaw` only about vertical; `off` keeps its orientation |
| `HAND_TURN_DEADBAND_DEG` | `5` | Turns of the hand smaller than this do nothing; larger ones count from here |
| `HAND_SCALE` | `1.0` | Arm metres per hand metre, for quick moves |
| `HAND_FINE_SCALE` | `0.4` | The share of `HAND_SCALE` that slow moves get; `1` turns fine moves off |
| `HAND_LEFT_YAW_DEG`, `HAND_RIGHT_YAW_DEG` | `0` | In `base` mode only: how far an arm's base is turned left of facing the workspace |
| `HAND_TABLE_Z_M` | `0.0` | The floor under each arm |
| `HAND_MAX_SPEED_M_S` | `0.5` | Target speed limit |
| `HAND_MAX_JOINT_RAD_S` | `1.5` | Joint speed limit in the IK |
| `HAND_CAMERA_HFOV_DEG` | `70` | The webcam's field of view: scales forward and back, and keeps a hand away from the image's centre from reading as turned |
| `HAND_SWAP_SIDES` | `0` | `1` for a webcam that mirrors its image (your hands drive the wrong arms) |
| `HAND_TELEOP_IDENTITY` | `hand-teleop` | Set `teleoperator` to stand in for the leader arms (not alongside them) |
| `HAND_MODEL_PATH` | | A local `hand_landmarker.task` instead of the download |

## How it works

`src/utils/hands.py` holds the pieces, all testable without a camera (`test/test_hands.py`): the
clutch (finger curl as tip-to-wrist over knuckle-to-wrist: about 1.8 straight, below 1.0 once the
tip folds back past its knuckle into the palm; closing below 1.0 and opening above 1.15), the palm's
position, the palm's turn (axes from the wrist and knuckles), 1€ filters, the tool-relative mapping
(or optional fixed base mapping), and the IK: `dream_ik`'s `ChunkIK` for the selected Metal, B601-DM, WidowX or YAM arm, one tick per call, the same
solver Dream-1's client uses (chapter 2), with the turn weighed lower than the place: at the
solver's own weights a hand tilted past what the arm can do from its rest pose drags the tool up to
11 cm.

The palm's position comes from its five rigid points (wrist and knuckles) alone, so a pinch never
moves the arm: shaped as MediaPipe's world landmarks have them, they are placed where they project
onto their pixels. Their size is the median of the last few seconds of sightings, held for the
length of a grip, because MediaPipe estimates its metric scale afresh every frame, and a few percent
of wander there is centimetres of depth at arm's length: enough to work the elbow back and forth
under a still hand. The filters run on every sighting, gripping or not, so a grip anchors to a settled
estimate rather than to one frame.

MediaPipe labels each hand as it appears in the image it is given, so detection runs on the raw
webcam frame and only the preview is mirrored. It reads each hand's 3D landmarks off a crop around
it, so they come turned as the hand looks along its own line of sight; turning them back by the angle
between that line and the lens's axis is what keeps a hand at the image's edge from reading as
turned by up to half the field of view.

A control change during an IK solve discards that pending action. The operator's
actions are ordinary Portal actions: the recorder archives them during a take, but
takes are still marked from the leader-arm teleop (`r`).
