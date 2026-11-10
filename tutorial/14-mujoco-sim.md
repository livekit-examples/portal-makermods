# 14. A cloud MuJoCo workspace with WidowX or YAM arms

Run two **Trossen WidowX AI follower** arms (default) or **I2RT YAM** arms
in MuJoCo, publish their state and four
cameras through Portal, and switch between browser teleop and a policy. No
physical robot or CAN device is opened. Use a separate LiveKit room from your
hardware sessions.

The follower models and wrist-camera mounts are vendored from
[Trossen's official MuJoCo assets](https://github.com/TrossenRobotics/trossen_arm_mujoco),
with their license and pinned revision in
[`assets/widowx`](../assets/widowx/PROVENANCE.md). The scene has a bare maple workbench in a simple room with a window and ceiling lights,
movable desk items, and textured RoboCasa household objects. This profile now uses **WidowX AI**, not the classic 250S;
recordings/checkpoints from the previous 250S profile are incompatible.

## 1. Start locally

Use Python 3.12, uv, Node 22.22+ and a recent LiveKit server that forwards
per-frame `user_timestamp` metadata (**tested with LiveKit 1.13.11**).
LiveKit Cloud also works. Older servers
can deliver visible video but no synchronized Portal observations; the UI then
stays at “Waiting for fresh observations” and disables control.

The simulator is pinned to **MuJoCo 4.0.0** in `pyproject.toml` and `uv.lock`,
including the cloud image. Version 4 installs `mujoco-core` and `mujoco-studio`
alongside the `mujoco` package. All four cameras now default to
[Filament](https://mujoco.readthedocs.io/en/4.0.0/programming/rendering.html),
using physically based workbench materials, soft shadows, ambient occlusion,
4× multisample antialiasing and neutral tone mapping. Texture assets and camera
calibration remain the same. Object contacts use full implicit integration
with a dense constraint Jacobian.

From the repository root:

```bash
uv sync --group sim --inexact
npm --prefix web/sim ci
npm --prefix web/sim run build
```

When upgrading an existing 3.x environment, reinstall the split packages together:

```bash
uv sync --group sim --inexact --reinstall-package mujoco \
  --reinstall-package mujoco-core --reinstall-package mujoco-studio
```

Restart running simulation and policy processes after upgrading: a process
that has already loaded MuJoCo retains its old library until it exits. Start
the simulator first, then Dream, and leave control unclaimed until fresh
observations arrive. The simulator's startup line prints its actual engine
version. Rebuild Docker images with `docker compose ... up --build` as below.

Set these in `.env` (see `.env.example`):

```dotenv
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_ROOM=portal-widowx-sim
PORTAL_PROFILE=widowx
```

**Check `.env.local` too:** this repo loads it after `.env` with override enabled,
so it can override inline shell variables, including `PORTAL_PROFILE`. Set
`PORTAL_PROFILE=widowx` there as well when using this checkout for simulation,
or use a separate simulation checkout. Every peer must use the same room/profile.

The dev key/secret are only for a local `livekit-server --dev`. Use your own
credentials with a remote server. Do not commit `.env` or `.env.sim`.

Start the server and simulator in separate terminals:

```bash
livekit-server --dev
```

```bash
uv run --group sim --inexact robot-sim
```

Open **http://127.0.0.1:8080**, enter the access code printed by `robot-sim`, and
wait for all four cameras. To choose a persistent access code, set
`SIM_WEB_TOKEN` before starting. Binding to a non-loopback interface requires
that variable; the server will not generate a public-server default.

Filament is selected by default; `--renderer filament` makes the choice explicit.
`--renderer` overrides `SIM_RENDERER`. The classic renderer remains available:

```bash
uv run --group sim robot-sim --arms yam --renderer filament
uv run --group sim robot-sim --renderer classic
```

MuJoCo 4.0's Filament backend uses OpenGL on macOS and Vulkan on Linux. On macOS,
use a graphical login session; no desktop viewer opens. The simulator starts a
spawned process so Filament runs on its main thread (Mac readback hangs on the
old background thread). Portal networking remains asynchronous in the parent.
The shipped 4.0.0 wheel requires importing Studio to register shader resources;
it may print an upstream duplicate `ReleasablePointer` class warning on macOS.

On Linux, install the Vulkan loader and NVIDIA graphics driver/ICD; the cloud
image includes the loader and requests graphics access from the Container Toolkit.
`MUJOCO_GL` does not select Filament's backend. For the **classic** fallback only,
set `MUJOCO_GL=egl` on headless Linux; leave it unset on macOS.

Verify both renderers and arm options (four cameras, camera orbit, readback
ownership, process startup/shutdown):

```bash
uv run --group sim --inexact python -m test.test_mujoco_render
```

Filament was verified locally on an Apple M4 Pro. A short 640×480 four-camera
sample took about 20 ms per physics/render/transfer tick; this is not a sustained
streaming benchmark or a cloud GPU guarantee. The Linux Vulkan path still needs
validation on your GPU instance. Higher resolutions cost more; watch the browser's
real-time factor. Renderer changes alter policy observations, so collect and
evaluate demonstrations with the renderer you intend to use for inference.

For a minimal cloud-only environment without the repo's hardware/training
dependencies, use `uv sync --only-group sim --no-install-project`, then run
`PYTHONPATH=src .venv/bin/python -m robot_sim` and `-m sim_policy`. Use a separate
virtual environment if you also use this checkout for hardware or ACT.

### Select I2RT YAM instead of WidowX

Stop the current simulation and close any recording before switching the arm
pair. The selection takes effect at process startup; it is not an in-place
browser reset. Start each peer in its own terminal, using the same LiveKit room:

```bash
uv run --group sim --inexact robot-sim --arms yam
uv run --inexact dream-1-tasks --profile yam
uv run --inexact recorder --profile yam
# Optional transport demo instead of Dream:
uv run --group sim --inexact sim-policy --profile yam
```

Use `--arms widowx` and `--profile widowx` to switch back. The simulator defaults
to `SIM_ARMS` when present, otherwise WidowX. The demo policy uses the same
fallback; explicit flags override `.env` and `.env.local`. The browser adopts
the selected profile automatically and shows the arm name and correct joint
labels. It rejects Dream handoff if the policy reports the other embodiment.
Reconnect the browser and restart policies/recorders with the matching profile.

The YAM option uses the six-axis YAM / YAM Pro model with a **linear_4310**
gripper and native D405 wrist camera from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/i2rt_yam).
Assets, license and revision are vendored in `assets/yam/`; this does not install
or open a physical YAM driver. It keeps the same workbench, props, four streams,
exercise resets, teleop controls and recording flow. It is not the 7-DOF or BIG
YAM variant.

| Profile | Arm joints per side | Portal gripper | Native carriage travel |
| --- | --- | --- | --- |
| `widowx` | `joint_0` … `joint_5` | 0 closed … 1 open | 0–44 mm |
| `yam` | `joint_1` … `joint_6` | 0 closed … 1 open | 0–47.5 mm |

YAM's custom Dream IK uses its own model-derived URDF and `grasp_site` TCP,
134.7 mm along flange Z. It is checked against the simulated chain, including
jaw direction, rather than sharing WidowX's IK or assuming a built-in YAM tool
offset matches this gripper. Model task success remains unverified. Keep
recordings and checkpoints separate for the two embodiments; the joint schemas,
kinematics and camera geometry differ. ACT uses `PORTAL_PROFILE=yam` with a YAM
checkpoint and dataset.

## 2. Teleoperate and hand off

- **Take control** explicitly selects this browser's Portal operator. Sliders
  command six joints and a normalized gripper for the selected arm. The other
  arm retains its targets. There is no automatic control-claim loop.
- **Hold** clears the gate and cancels outstanding targets at the measured
  pose. Escape, leaving the tab, browser disconnect, and a 750 ms command
  timeout also release browser control. A network failure may take up to the
  command timeout to stop an outstanding move.
- Drag the **Scene** video to orbit; scroll to zoom. This is one shared,
  server-rendered spectator camera. It does not move the overhead or wrist
  cameras. Other connected viewers see the same spectator view.
- **Return home** releases control and ramps targets toward the startup joint
  pose. **Reset scene** releases control and restores both arms and objects,
  incrementing the episode counter. Choose **Reset setup** before resetting:
  **Workbench · household objects**. Reset never automatically resumes an operator.
  These simulation commands do not call physical robot park/power RPCs.

Webcam hand control is also available on your local machine:

```bash
uv run --group hands hand-teleop --profile widowx
# For the YAM scene, use --profile yam instead.
```

Use the simulator's LiveKit room, press `c` to take control, then curl middle,
ring and pinky to engage each arm. `p` releases control and returns the simulated
arms home. See [chapter 11](11-hand-teleop.md) for gestures and camera settings.
Motion defaults to each gripper's POV: toward the webcam advances along its
fingers, and wrist twist rolls it regardless of hand orientation. Use `--frame base`
to restore fixed robot-base directions.

To exercise the handoff:

```bash
uv run --group sim --inexact sim-policy
```

Select **Demo · waist sweep**, then **Hand off**. The demo performs a small
sinusoidal waist sweep around the pose observed when it receives control.
Press **Take control** to interrupt it; select it again to resume from a new
observed pose. It is a deterministic transport demo, not inference or a task
success benchmark. It starts idle and never claims itself.

For Dream-1, set `DREAM_API_KEY` in the policy process environment and run:

```bash
uv run --inexact dream-1-tasks --profile widowx
```

The explicit flag overrides `PORTAL_PROFILE`, including `.env.local`. The
operator prints `widowx ready` when connected. Without a flag it reads
`PORTAL_PROFILE` (falling back to ReBot when unset). The sim's minimal
`--only-group sim` environment does not include Dream; use the regular repo
environment above, or the optional cloud image below.

Wait for `widowx: receiving timestamped observations (14 state fields, 4 camera
frames)` before handing off. If `missing-timestamp` warnings persist, restart
the receiving peer after the robot is running, using the explicit profile
flag. Confirm all peers use this checkout's environment and the same room.
The simulator already enables frame metadata through Portal and supplies the
same capture timestamp for state and cameras; do not replace missing capture
timestamps with receive time. For a self-hosted server, also check the server
version described above. Browser video alone does not verify timestamps.

Enter an instruction (1–200 characters) in **Demonstration**. It syncs after a
brief typing pause; Enter or leaving the field sends it immediately. The
field's initial value is also sent on connection.
The server keeps the latest task and sets it on `policy-dream-1-tasks` when
that operator connects or reconnects. The status beneath the field reports
queued, applied, or RPC errors. A new task pauses Dream if it is driving;
it never starts motion. Select `policy-dream-1-tasks` and click **Hand off**
to confirm the current field's task and explicitly call `dream.resume`.
A wrong-profile operator is rejected with instructions to restart it.

Dream receives the overhead and both native wrist cameras, both six-joint
states in URDF radians, and normalized grippers. Its EEF predictions pass
through custom-arm IK from `assets/widowx/dream_arm.json` and the matching
URDF; the TCP is Trossen's MJCF `ee_site`. The spectator view is excluded.
The SDK's warning about missing physical TCP calibration is expected for
this simulation-derived model. No physical-arm calibration is claimed.
`p` in this operator homes the simulation; `c` explicitly takes/releases
control. A task change or handoff discards old inference chunks.

Dream-1 inference runs on Dream Labs' service; its SDK runs IK locally. A
correct embodiment mapping does not establish task success in this scene.
The repository tests transport and kinematics without calling paid inference.

Alternatively, use an ACT checkpoint trained for this embodiment:

```bash
PORTAL_PROFILE=widowx uv run --inexact act \
  --checkpoint outputs/widowx/checkpoints/last/pretrained_model \
  --dataset-root data/lerobot/widowx \
  --task 'Put the apple in the tray'
```

Select `policy-act` in the browser. ACT still uses dataset `meta/info.json`
order and validates the schema. Existing Metal, ReBot, and classic WidowX 250S
checkpoints are incompatible with this WidowX AI profile. For a new policy, follow `sim_policy.py`'s Portal
Operator flow, replace the action computation, run blocking inference off the
event loop, and discard results when control changes during inference.

### Workbench exercises

The centered overhead camera follows Trossen's stationary bimanual camera pose:
about 1.05 m above the work surface, tilted toward the shared workspace. The
native wrist cameras remain on the arms. The spectator view looks across the
bench into an office with a desk, monitor, chair, shelves and window. Office
furniture is visual scenery; manipulation collisions are on the workbench,
floor and task props.

Camera clipping is fixed at 5 mm near / 20 m far after model composition.
MuJoCo's XML clipping values are relative to scene extent; using an unadjusted
near value with room-sized scenery can slice through the nearby finger meshes
and leave jagged triangles in wrist views.

The workbench includes a small vendored set from [RoboCasa's official asset
release](https://huggingface.co/datasets/robocasa/robocasa-assets/tree/1b92c3d02ca4354984fec961357db0bff7b32166):
a mug, blue water bottle, scissors, tray, apple, banana, lemon, drink can, sponge,
soap bar, spoon and shallow bowl. Their original visual meshes and textures are
separate from contact geometry; decomposed collision pieces preserve the hollow
mug, bowl and tray. Both arm pairs load the same workbench layout.

This approximately 13 MB subset ships in `assets/robocasa`; no RoboCasa or
robosuite dependency, extra download, or new setup command is required. Asset
revision, checksums, CC-BY-4.0 attribution and adaptation details are in
[`assets/robocasa/PROVENANCE.md`](../assets/robocasa/PROVENANCE.md). Masses and
friction are illustrative. All objects are rigid: scissors do not cut, the bottle
cap stays fixed, sponges do not squeeze, and liquids are not modeled.

Try these instructions in **Demonstration**, then explicitly hand off to Dream:

- “Put the apple in the blue tray.”
- “Place the lemon in the bowl.”
- “Move the banana next to the apple.”
- “Place the USB drive next to the red notebook.”
- “Put the scissors in the blue tray.”
- “Place the mug next to the water bottle.”
- “Move the sponge next to the soap.”
- “Put the spoon beside the bowl.”

The colored toys and green mat have been removed. Household props sit directly
on the wooden tabletop alongside the USB drive, screwdriver and notebook.
Everything is movable except the fixed tray. Reset restores this household layout.

Object poses stay local to the simulator; Dream receives only robot state and
camera observations. Reset restores every prop to its initial position.

Tests cover textured asset integrity, settling/reset with both arm pairs, camera
framing, and a diagnostic probe falling inside the mug, bowl and tray. These prompts are exercises
to evaluate and collect corrections for; autonomous grasping and model success
on these tasks have not been established. Start with simple sorting to check
perception and control first.

## 3. Record demonstrations

Keep recording in a separate process:

```bash
uv run --inexact recorder --profile widowx
```

The flag overrides `PORTAL_PROFILE` even in `.env.local`. The recorder's
startup log should show 14 state/action fields and all four camera tracks.
Use the regular repository environment for the recorder; the minimal sim
image does not include Rerun's archive dependencies. It can run on another
host with access to the same LiveKit room. See [chapter 7](07-hitl-recording.md)
for ReBot and Metal recording commands.

The same Demonstration task labels the take. Enter it, click **Start take**, teleoperate or hand off, and
click **End take**. **Discard** rejects a take. The UI only reports a marker as
recorded after a recorder acknowledges it. Reset and home are disabled during
that browser's take. Close the take before a reset from any client; reset does
not create a training episode boundary automatically. Closing a browser with
an open take sends a discard marker when its connection still permits it.

Use [chapter 7](07-hitl-recording.md) to convert the archive. Both archive
timelines and Portal's action send timestamps are preserved. Browser actions
carry the timestamp of the synchronized observation delivered to the browser
bridge. This is not a measurement of the exact video frame currently displayed
by the browser's independent WebRTC playback pipeline.

## 4. Run on a cloud NVIDIA GPU

Use a Linux x86-64 GPU instance with NVIDIA drivers, Docker, the NVIDIA
Container Toolkit, and Docker Compose supporting `gpus: all`. The image uses
headless Filament/Vulkan rendering by default, with classic/EGL available via
`SIM_RENDERER=classic`; there is no X server, VNC, or desktop streaming.
MuJoCo physics remains on CPU; the GPU renders cameras (and can run a separate
policy). This example does not claim GPU physics or zero-copy video encoding.

Create `.env.sim` containing only the simulation's settings:

```dotenv
LIVEKIT_URL=wss://YOUR_PROJECT.livekit.cloud
LIVEKIT_API_KEY=YOUR_KEY
LIVEKIT_API_SECRET=YOUR_SECRET
LIVEKIT_ROOM=portal-widowx-sim
PORTAL_PROFILE=widowx
SIM_WEB_TOKEN=YOUR_LONG_RANDOM_ACCESS_CODE
```

Use `python -c 'import secrets; print(secrets.token_urlsafe(32))'` to generate
an access code. Both the server and browser must reach `LIVEKIT_URL`; set
`LIVEKIT_BROWSER_URL` if the browser needs a different address. `localhost`
in a container refers to that container, not a LiveKit server on your laptop.

```bash
docker compose -f deploy/compose.sim.yaml --profile demo up --build
```

This starts the simulator and an idle demo policy. Omit `--profile demo` to run
only the simulator. The simulation image installs only the `sim` dependency group, serves
its own compiled browser bundle, and includes the models; it needs no package
CDN or asset downloads at runtime. LiveKit remains an external service.

To run Dream-1 instead of the sweep demo, add `DREAM_API_KEY` to `.env.sim`:

```bash
docker compose -f deploy/compose.sim.yaml --profile dream up --build
```

The optional `dream-policy` image installs the `sim-dream` group (Dream's SDK
and IK dependencies) and reads `PORTAL_PROFILE` (the image defaults to `widowx`). It joins idle;
set the browser task and click **Hand off**. It needs outbound access to Dream
Labs. Shut down with the same command using `down` instead of `up --build`.
For YAM, set **both** `SIM_ARMS=yam` and `PORTAL_PROFILE=yam` in `.env.sim`
and rebuild/recreate the services. `SIM_ARMS` selects the simulator and demo;
`PORTAL_PROFILE` selects Dream and the recorder. Both arm assets are included
in the same image. Switching back uses `widowx` for both values.
The cloud Docker/NVIDIA path still requires validation on a GPU host.

The HTTP port is bound to the VM's loopback address. From your laptop:

```bash
ssh -L 8080:127.0.0.1:8080 USER@GPU_HOST
```

Open **http://127.0.0.1:8080** locally and enter `SIM_WEB_TOKEN`. This carries the
UI/control WebSocket over SSH; camera media travels through LiveKit. For a
shared public UI, put an HTTPS reverse proxy in front of port 8080, preserve
`Host`, forward WebSocket upgrades, and set `X-Forwarded-Proto: https`.
Credentials are exchanged only with the same-origin server. The browser gets
a subscribe-only LiveKit token; API secrets never leave the server. The access
code grants shared simulation control, not separate per-user permissions.

Shutdown:

```bash
docker compose -f deploy/compose.sim.yaml --profile demo down
```

## 5. Rates, wire contract, and implementation

`profiles.widowx` and `profiles.yam` in `portal.yaml` define the simulation wire
contracts. All peers load the selected profile
through `utils.common.portal_config`. It declares 14 joint/gripper values and
`overhead_camera`, `left_wrist_camera`, `right_wrist_camera`, `scene_camera`.
WidowX arm fields are `left_joint_0.pos` through `left_joint_5.pos` and their
right-arm equivalents, plus each gripper. Joint values are radians; grippers
are **0 closed .. 1 open**, mapped to the native **0–0.044 m** carriage travel. Never use Metal's millimetre gripper units
with either simulation profile. YAM uses `joint_1..6` and 0–0.0475 m carriage travel.

The default is 240 Hz fixed-step physics, in eight substeps per 30 Hz
observation. Filament runs physics and rendering on a dedicated process main
thread; a serialized worker in the parent transfers commands and RGB arrays
through a pipe. Classic rendering uses that worker directly. The asyncio thread
handles networking. Joint targets are clamped to actuator limits and ramped
at 1.5 rad/s (grippers: 1.5 normalized openings/s). Step and render are serialized
so all four images and state describe the same physics snapshot. MuJoCo 4.0's
Filament backend supports one pixel read request at a time, so cameras render
sequentially without stepping physics between them. Rendering
and RGB readback take time: if they exceed the 33 ms budget, simulation slows
rather than skipping physics steps or accumulating a queue. Watch the UI's
real-time factor. Try `--width 320 --height 240` if necessary.

The bridge creates a Python `Operator` per browser WebSocket. Browser input
passes through `Operator.send_action` and the robot's active-operator gate;
it never writes simulated joints directly. State and cameras use one Portal
clock stamp per snapshot; simulation time resets separately. Handoff/reset
invalidate queued actions and old observation anchors. Policies must anchor
actions to fresh observations (within 750 ms for this example).

- `src/robot_sim.py`: robot lifecycle, control invalidation, watchdog, simulation loop.
- `src/utils/mujoco_sim.py`: model composition, actuators, physics, cameras.
- `src/utils/sim_render.py`, `sim_process.py`: Filament cameras and process lifecycle.
- `assets/workbench/office.xml`: workbench, office, interactive desk items on the bare tabletop.
- `src/utils/sim_objects.py`, `assets/robocasa/`: offline composition of textured props.
- `src/utils/sim_web.py`: access code, video viewer tokens, browser operator bridge.
- `src/sim_policy.py`: minimal handoff demonstration policy.
- `web/sim/`: browser console, bundled with Vite.
- `deploy/Dockerfile.sim`, `deploy/compose.sim.yaml`: NVIDIA EGL runtime.

Run the model/control and HTTP authentication checks without a display:

```bash
uv run --group sim --inexact python -m unittest test.test_mujoco_sim test.test_workbench test.test_robocasa_objects test.test_yam_sim test.test_sim_web test.test_dream1_widowx test.test_sim_tasks
```

For the browser task RPC test (no inference or GPU), start a disposable
LiveKit dev server and run:

```bash
SIM_TASK_TEST_LIVEKIT=ws://127.0.0.1:7880 uv run --inexact python -m unittest test.test_sim_task_rpc
```

For the motion transport test, start `robot-sim` and `sim-policy` against a recent
LiveKit server, then set `SIM_TEST_URL` and `SIM_WEB_TOKEN` for that **test**
instance and run `python -m test.test_sim_handoff`. It takes control and resets
the simulated scene; do not point it at an active demonstration session.

---

Previous: [13. Single ReBot](13-rebot.md) · [Tutorial index](../README.md#the-tutorial)
