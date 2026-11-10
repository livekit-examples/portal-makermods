# Office workbench assets

`office.xml` is an authored MuJoCo scene for this tutorial. It contains the
1.50 × 0.96 m maple workbench (0.78 m high), camera gantry, plain room with a window and ceiling lights,
interactive desk items and textured household objects on the bare wood surface.
The work surface is world Z=0 so the existing arm frames remain unchanged.
Room details use non-colliding visual geometry; the bench, floor, tools,
desk items and imported props participate in physics.

The maple texture is the unmodified `assets/maple_tabletop.png` from
[Trossen's assets at revision 77aba5d](https://github.com/TrossenRobotics/trossen_arm_mujoco/tree/77aba5d32654f17945f139e63310e7a3ac2cd4ab).
The overhead camera position/quaternion follows that revision's
`assets/stationary_ai/stationary_ai.xml`, with a 65° vertical field of view.
See [the bundled BSD-3-Clause license](../widowx/LICENSE).
The office and desk-item geometry are authored here. The composed scene also
loads textured fruit, cleaning items, kitchenware, a drink can, mug, water bottle,
scissors and tray from
[`assets/robocasa`](../robocasa/PROVENANCE.md). Those assets retain their own
CC-BY-4.0 attribution and license; no model or texture download is needed at runtime.

All household objects are free bodies except the fixed tray. Their original
visual meshes/textures and contact geometry are retained. The colored toy
shapes, green mat, cable and switch have been removed. Props rest directly on
the wood at Z=0. See [tutorial 14](../../tutorial/14-mujoco-sim.md) for task suggestions.

Filament is the default camera renderer. Authored materials include metallic and
roughness values; `filament.*` settings enable soft shadows, ambient occlusion
and neutral tone mapping. The renderer uses 4× MSAA. Classic rendering retains
the original Phong material values and ignores the Filament custom settings.

The background desk, computer, chair, shelving, pegboard and plant are removed
to keep attention on the workbench.
