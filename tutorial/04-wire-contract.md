# 4. The shared schema

This chapter explains the configuration used in the walkthrough. Open
[`portal.yaml`](../portal.yaml); keep the same file on every host.

## 1. Read the fields

The schema declares three camera tracks and fourteen positions: six joints
and one gripper per arm. State reports measured positions; actions request
target positions. Joints use radians. The Metal gripper uses its stroke in millimetres,
0 closed to 80 open; see
[`metal.py`](../src/utils/metal.py).

The schema sets a 30 Hz rate and H264 video. Keep these defaults for the walkthrough.
All three declared cameras must publish before operators can anchor actions
to a complete camera view.

## 2. Load the same schema for every role

```python
robot_cfg = portal_config(RobotConfig, room, "metal")
operator_cfg = portal_config(OperatorConfig, room, "metal")
observer_cfg = portal_config(ObserverConfig, room)
```

Teleop, Dream-1, and ACT all use `OperatorConfig`. Mismatched schemas produce
fingerprint warnings and dropped packets, so deploy schema changes to every
peer together.

`utils.common.portal_config` loads `portal.yaml`, removes the `profiles` block,
and passes the selected contract to Portal. The top-level fields are the Metal
rig; `profiles.rebot` replaces state, action, and videos with seven values and
two cameras. `robot-rebot` and `dream-1-tasks` explicitly select `rebot`.
Recorder, task monitor, and ACT use `PORTAL_PROFILE` (default `metal`); set it to
`rebot` for that room. ReBot grip is normalized **0..1**, unlike Metal's 0..80 mm.

`profiles.widowx` is the [dual-arm MuJoCo simulation](14-mujoco-sim.md):
14 named WidowX positions (radians, grip 0..1) and four VP8 cameras, including
a movable spectator view. `robot-sim` and `sim-policy` select it explicitly;
use `PORTAL_PROFILE=widowx` on the recorder and ACT hosts. The named joints
intentionally differ from Metal so an incompatible checkpoint fails validation.
Do not load this extended YAML directly with Portal's `from_yaml_file`.

## 3. Preserve training order

ACT reads state and action order from the training dataset's `meta/info.json`
and checks the names against the schema. Do not alphabetize fields: model
tensors and normalization statistics use the dataset's order.

## Transport defaults

H264 carries video over WebRTC media. State and actions use unreliable packets
so a lost packet does not hold up newer readings. Keypoints and control messages
use reliable delivery.

Other video formats are available, but keep H264 unless your application needs
a different transport.

---

Previous: [3. Train ACT and improve it with corrections](03-train-act.md) · [Tutorial index](../README.md#the-tutorial) · Next: [5. Robot loop](05-robot-loop.md)
