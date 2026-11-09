"""Map the single ReBot to Dream-1's bimanual model interface.

Only the right output reaches Portal. The absent left arm has a fixed model
pose and a black wrist view; these placeholders never enter recordings.
"""
import json
from pathlib import Path

import dream
import numpy as np

from policies.dream_1.run import embodiment, encode_frame
from utils.rebot import clamp, fields_from_config

CAMERAS = {"overhead": "overhead_camera", "wrist_r": "arm_camera"}


class RebotMapping:
    def __init__(self, cfg, arm_file: str):
        self.fields = fields_from_config(cfg)
        self.arm_file = str(Path(arm_file).expanduser().resolve())
        spec = json.loads(Path(self.arm_file).read_text())
        rest = np.asarray(spec["rest"], dtype=float)
        if rest.shape != (6,) or not np.isfinite(rest).all():
            raise ValueError("REBOT_ARM_FILE must describe six joints in URDF order")
        self.left_rest = rest.tolist()
        self.blank_wrist = encode_frame(np.zeros((176, 320, 3), dtype=np.uint8))

    def policy_config(self, checkpoint, row_hz):
        return {"checkpoint": checkpoint, "embodiment": embodiment(),
                "ik": {"arm": self.arm_file, "head": "eef"}, "action_hz": row_hz}

    def to_model(self, state):
        values = np.asarray([state[name] for name in self.fields], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("ReBot state must be finite")
        return values

    def to_robot(self, row):
        return dict(zip(self.fields, clamp(row)))

    def observation(self, state, task):
        row = self.to_model(state)
        return {"prompt": task, "state.right_arm": row[:6].tolist(), "state.right_grip": row[6:].tolist(),
                "state.left_arm": self.left_rest, "state.left_grip": [0.0]}


def chunk_rows(reply):
    """Only the physical arm's chunk: [rows, six joints + normalized grip]."""
    try:
        rows = np.asarray(reply["q_r_chunk"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise dream.DreamActionError("missing or malformed ReBot right-arm chunk") from exc
    if rows.ndim != 2 or rows.shape[1] != 7 or len(rows) == 0 or not np.isfinite(rows).all():
        raise dream.DreamActionError(f"ReBot chunk has shape {rows.shape}; expected nonempty [rows, 7]")
    return rows
