"""A selectable dual-arm workbench. One worker owns physics and rendering."""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from utils.common import ROOT
from utils.sim_objects import add_objects
from utils.sim_render import FilamentRenderer, RENDERERS

MODEL_DIR = ROOT / "assets" / "widowx"
ARMS = {"widowx": "Trossen WidowX AI", "yam": "I2RT YAM"}
SETUPS = {"workbench": "Workbench · household objects"}


def validate_action(values, limits):
    if set(values) != set(limits):
        raise ValueError("action must contain every simulated field exactly once")
    result = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"non-finite or non-numeric action: {name}")
        result[name] = float(np.clip(value, *limits[name]))
    return result


def build_model(model_dir: Path | None = None, *, arms="widowx"):
    import mujoco

    if arms not in ARMS:
        raise ValueError(f"unknown simulation arms: {arms}")
    model_dir = model_dir or ROOT / "assets" / arms
    scene = mujoco.MjSpec.from_file(str(ROOT / "assets" / "workbench" / "office.xml"))
    add_objects(scene)
    for side, y in (("left", .3), ("right", -.3)):
        path = "wxai/wxai_follower.xml" if arms == "widowx" else "yam_linear_4310.xml"
        arm = mujoco.MjSpec.from_file(str(model_dir / path))
        # Keyframe arrays describe one arm, not the composed scene. Reset below sets named joints.
        for key in list(arm.keys):
            arm.delete(key)
        # The composite scene uses full implicit integration for stable object contacts.
        arm.option.integrator = scene.option.integrator
        arm.camera("cam" if arms == "widowx" else "wrist_cam").name = "wrist_camera"
        if arms == "widowx":
            arm.actuator("left_gripper").name = "gripper"
        else:
            # Distinct from Metal's joint1..6 and WidowX's joint_0..5 wire contract.
            for index in range(1, 7):
                name = f"joint_{index}"
                arm.joint(f"joint{index}").name = name
                actuator = arm.actuator(f"joint{index}")
                actuator.name = actuator.target = name
        frame = scene.worldbody.add_frame(pos=[-.28, y, 0])
        scene.attach(arm, prefix=side + "/", frame=frame)
    model = scene.compile()
    # MuJoCo scales clip planes by scene extent. Room-sized scenery must not
    # push the near plane through the fingers in the native wrist cameras.
    model.vis.map.znear = .005 / model.stat.extent  # 5 mm
    model.vis.map.zfar = 20. / model.stat.extent  # 20 m; retain useful depth precision
    return model


class Simulation:
    def __init__(self, cfg, *, width=640, height=480, physics_hz=240, render=True, arms="widowx",
                 renderer="filament"):
        import mujoco

        if not 1 <= cfg.fps <= physics_hz or physics_hz % cfg.fps:
            raise ValueError("physics_hz must be a multiple of the observation fps")
        if not 64 <= width <= 1280 or not 64 <= height <= 960:
            raise ValueError("render size must be within 64x64 .. 1280x960")
        if renderer not in RENDERERS:
            raise ValueError(f"unknown renderer: {renderer}")
        self.render_backend = renderer
        self.mj = mujoco
        self.version = mujoco.mj_versionString()
        self.arms = arms
        self.model = build_model(arms=arms)
        self.model.opt.timestep = 1 / physics_hz
        self.data = mujoco.MjData(self.model)
        self.substeps = physics_hz // cfg.fps
        self.fields = [field.name for field in cfg.action_schema]
        if set(self.fields) != {field.name for field in cfg.state_schema}:
            raise ValueError("Simulation state and action fields must match")
        self.bindings = {}
        self.limits = {}
        for field in self.fields:
            side, joint = field.removesuffix(".pos").split("_", 1)
            actuator = self.model.actuator(f"{side}/{joint}")
            joint_id = self.model.actuator_trnid[actuator.id, 0]
            address = int(self.model.jnt_qposadr[joint_id])
            lo, hi = map(float, actuator.ctrlrange)
            grip = joint == "gripper"
            if grip:
                lo = 0.  # YAM allows negative squeeze travel; Portal 0 means physically closed.
            self.bindings[field] = (actuator.id, address, lo, hi, grip)
            self.limits[field] = [0., 1.] if grip else [lo, hi]
        if len(self.bindings) != self.model.nu:
            raise ValueError("portal.yaml must describe all simulated actuators")
        self.cameras = list(cfg.video_tracks)
        for name in self.cameras:
            if name != "scene_camera":
                self.model.camera(name.replace("_wrist_camera", "/wrist_camera"))
        self.renderer = None
        if render:
            self.renderer = (FilamentRenderer(self.model, self.cameras, width=width, height=height)
                             if renderer == "filament" else mujoco.Renderer(self.model, height=height, width=width))
        self.scene_camera = mujoco.MjvCamera()
        self.scene_camera.lookat[:] = [.02, 0, .18]
        self.scene_camera.distance = 2.45
        self.scene_camera.azimuth = 135
        self.scene_camera.elevation = -25
        self.view_options = mujoco.MjvOption()
        self.view_options.geomgroup[3] = 0  # show visual meshes, hide collision meshes
        self.episode = 0
        self.reset()

    def state(self):
        values = {}
        for name, (_, address, lo, hi, grip) in self.bindings.items():
            value = float(self.data.qpos[address])
            values[name] = float(np.clip((value - lo) / (hi - lo), 0, 1)) if grip else value
        return values

    def validate_action(self, values):
        return validate_action(values, self.limits)

    def hold(self):
        self.target = self.validate_action(self.state())
        self.commanded = self.target.copy()
        self._write_controls()

    def reset(self, setup=None):
        setup = setup or getattr(self, "setup", "workbench")
        if setup not in SETUPS:
            raise ValueError("unknown scene setup")
        self.setup = setup
        self.mj.mj_resetData(self.model, self.data)
        home = {"joint_0": 0., "joint_1": 1.1, "joint_2": 1.2, "joint_3": -.65,
                "joint_4": 0., "joint_5": 0., "left_carriage_joint": .03, "right_carriage_joint": .03}
        if self.arms == "yam":
            home = dict(zip((f"joint_{i}" for i in range(1, 7)), (0., 1.1, 1.2, -.65, 0., 0.)))
            home.update(left_finger=.03, right_finger=.03)
        for side in ("left", "right"):
            for name, value in home.items():
                self.data.joint(f"{side}/{name}").qpos[0] = value
        self.mj.mj_forward(self.model, self.data)
        self.hold()
        self.home = self.target.copy()
        self.episode += 1

    def set_view(self, view):
        bounds = {"azimuth": (-360, 360), "elevation": (-85, -5), "distance": (.5, 3)}
        if set(view) != set(bounds):
            raise ValueError("view needs azimuth, elevation, distance")
        for key, (low, high) in bounds.items():
            value = view[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("invalid camera value")
            setattr(self.scene_camera, key, float(np.clip(value, low, high)))

    def _write_controls(self):
        for name, (actuator, _, lo, hi, grip) in self.bindings.items():
            value = self.commanded[name]
            self.data.ctrl[actuator] = lo + value * (hi - lo) if grip else value

    def tick(self, action=None, *, hold=False, reset=False, home=False, view=None, setup=None):
        if reset:
            self.reset(setup)
        elif hold:
            self.hold()
        if home:
            self.target = self.home.copy()
        if action is not None:
            self.target = self.validate_action(action)
        if view is not None:
            self.set_view(view)
        expected_time = self.data.time + self.substeps * self.model.opt.timestep
        for _ in range(self.substeps):
            # Position targets ramp at 1.5 rad/s (gripper: 1.5 normalized openings/s).
            for name in self.fields:
                delta = self.target[name] - self.commanded[name]
                self.commanded[name] += float(np.clip(delta, -1.5 * self.model.opt.timestep,
                                                    1.5 * self.model.opt.timestep))
            self._write_controls()
            self.mj.mj_step(self.model, self.data)
        if (not math.isclose(self.data.time, expected_time, abs_tol=1e-8)
                or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()):
            raise RuntimeError("MuJoCo produced invalid state or automatically reset after instability")
        # mj_step's derived transforms can lag the final integration step.
        self.mj.mj_forward(self.model, self.data)
        state = self.state()
        frames = {}
        if self.renderer and self.render_backend == "filament":
            frames = self.renderer.render(self.data, self.scene_camera, self.view_options)
        elif self.renderer:
            for name in self.cameras:
                camera = self.scene_camera if name == "scene_camera" else name.replace("_wrist_camera", "/wrist_camera")
                self.renderer.update_scene(self.data, camera=camera, scene_option=self.view_options)
                frames[name] = self.renderer.render().copy()
        return state, frames, {"sim_time": self.data.time, "episode": self.episode,
                               "setup": self.setup, "renderer": self.render_backend}

    def close(self):
        if self.renderer:
            self.renderer.close()
