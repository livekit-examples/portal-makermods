"""Dream's custom-arm IK for simulated bimanual workbenches.

Joints are URDF radians, grips are 0 closed .. 1 open. Both wrist views are real;
the spectator camera is never sent to inference. No Metal units or ReBot padding.
"""
import xml.etree.ElementTree as ET

import dream
import numpy as np

from policies.dream_1.run import CAMERAS, embodiment
from utils.common import ROOT


class SimMapping:
    cameras = CAMERAS
    blank_frames = {}
    arm_count = 2

    def __init__(self, cfg, profile):
        self.arm_file = ROOT / "assets" / profile / "dream_arm.json"
        self.profile = profile
        joints = ET.parse(self.arm_file.with_suffix(".urdf")).getroot().findall("joint")
        names = [joint.get("name") for joint in joints] + ["gripper"]
        # Model order follows the kinematic chain, independently of YAML field order.
        self.fields = [f"{side}_{name}.pos" for side in ("left", "right") for name in names]
        if (set(self.fields) != {f.name for f in cfg.state_schema}
                or set(self.fields) != {f.name for f in cfg.action_schema}):
            raise ValueError(f"Dream {profile} requires the {profile} Portal profile")
        if set(self.cameras.values()) - set(cfg.video_tracks):
            raise ValueError("Dream simulation requires overhead and both wrist cameras")
        limits = [[float(j.find("limit").get(key)) for key in ("lower", "upper")] for j in joints]
        self.limits = np.asarray((limits + [[0., 1.]]) * 2)

    def policy_config(self, checkpoint, row_hz):
        return {"checkpoint": checkpoint, "embodiment": embodiment(),
                "ik": {"arm": str(self.arm_file), "head": "eef"}, "action_hz": row_hz}

    def to_model(self, state):
        row = np.asarray([state[name] for name in self.fields], dtype=float)
        if not np.isfinite(row).all():
            raise ValueError("Simulated state must be finite")
        return row

    def to_robot(self, row):
        row = np.asarray(row, dtype=float)
        if row.shape != (14,) or not np.isfinite(row).all():
            raise ValueError("Simulated action must have 14 finite values")
        return dict(zip(self.fields, np.clip(row, self.limits[:, 0], self.limits[:, 1]).tolist()))

    def observation(self, state, task):
        row = self.to_model(state)
        return {"prompt": task, "state.left_arm": row[:6].tolist(), "state.left_grip": row[6:7].tolist(),
                "state.right_arm": row[7:13].tolist(), "state.right_grip": row[13:].tolist()}

    @staticmethod
    def chunk_rows(reply):
        try:
            left, right = (np.asarray(reply[key], dtype=float) for key in ("q_l_chunk", "q_r_chunk"))
        except (KeyError, TypeError, ValueError) as exc:
            raise dream.DreamActionError("missing or malformed simulated arm chunks") from exc
        if (left.shape != right.shape or left.ndim != 2 or left.shape[1] != 7 or len(left) == 0
                or not np.isfinite([left, right]).all()):
            raise dream.DreamActionError("Simulation requires matching nonempty [rows, 7] chunks for both arms")
        return np.concatenate([left, right], axis=1)
