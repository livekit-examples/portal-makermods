# Trossen WidowX AI follower model

`wxai/wxai_follower.xml`, `meshes/`, `LICENSE`, and `README.md` are unmodified
files (only the referenced meshes are included) from
[TrossenRobotics/trossen_arm_mujoco](https://github.com/TrossenRobotics/trossen_arm_mujoco/tree/77aba5d32654f17945f139e63310e7a3ac2cd4ab),
commit `77aba5d32654f17945f139e63310e7a3ac2cd4ab` (also in `REVISION`).
`maple_tabletop.png` is copied unchanged from upstream `assets/maple_tabletop.png`
at that same revision. The included BSD-3-Clause license applies to those files
and the derived URDF.

`src/utils/mujoco_sim.py` composes two follower arms with separate prefixes,
renames `cam` to `wrist_camera` and the `left_gripper` actuator to `gripper`,
and attaches them to `assets/workbench/office.xml`. Native wrist cameras,
meshes, inertias, collision geometry, joints, finger coupling, gravity
compensation and actuator gains are retained. Home is a raised working pose,
not the upstream zero pose. This is WidowX AI, not classic WidowX 250S.

`dream_arm.urdf` is the six-joint kinematic chain derived from that MJCF:
origins, axes, order, position limits and effort limits are retained, with a
1.5 rad/s velocity cap. `dream_arm.json` selects it for Dream's custom-arm IK.
The TCP at +0.156 m on link_6 is the upstream `ee_site`; approach is +X,
jaw is +Y. Each arm uses its own base frame. This TCP is from simulation,
not physical calibration, so Dream's physical-calibration warning is expected.
The FK test compares this URDF to both simulated arms at several poses.

No model download is needed at runtime.
