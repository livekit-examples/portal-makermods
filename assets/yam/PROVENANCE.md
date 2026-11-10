# I2RT YAM with linear_4310 gripper

`yam_linear_4310.xml`, its referenced meshes in `assets/`, `README.md`,
`CHANGELOG.md` and `LICENSE` are copied unmodified from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/i2rt_yam),
revision `98a70c569c4c1b6da467a20012f766ee1f8dc521` (also in `REVISION`).
The upstream model is derived from I2RT's SDK v1.3.6 and is MIT licensed;
the bundled license covers these assets and the derived kinematic URDF.
Only meshes referenced by this variant are included. No runtime download or
I2RT hardware SDK is needed.

The simulator composes two six-axis YAM / YAM Pro arms with native linear_4310
grippers and D405 camera mounts into the existing workbench. It renames
`joint1..6` and their actuators to `joint_1..6`, and `wrist_cam` to
`wrist_camera`, then prefixes the copies with `left/` and `right/`.
It removes the single-arm keyframe and uses a raised working pose. The scene's
full implicit integrator replaces the model's implicitfast setting. Meshes,
collision geometry, inertias, native camera calibration, gravity compensation,
actuator gains and finger coupling are otherwise retained.

Portal grip is 0 closed .. 1 open, mapping to **0 .. 0.0475 m** finger travel.
The native model's negative squeeze range is not exposed as negative Portal
openings. Joint angles keep the model's native signs and limits. The YAM
profile deliberately differs from both WidowX joint_0..5 and Metal joint1..6.

`dream_arm.urdf` is the mesh-free six-joint chain derived from the compiled
MJCF's body transforms, joint axes and ranges. `dream_arm.json` uses `arm` as
the base and `link_6` as the flange; TCP is the native `grasp_site` at +0.1347 m
along flange Z, approach is +Z, and jaw separation is along X. The FK test
checks this chain, TCP and jaw against both composed simulated arms. These
are simulation geometry and camera calibration, not physical calibration or
validation of Dream-1 task success.
