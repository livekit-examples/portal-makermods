"""Compose the small, offline RoboCasa prop collection into the workbench."""
import json

from utils.common import ROOT

ASSET_DIR = ROOT / "assets" / "robocasa"


def add_objects(scene):
    import mujoco

    manifest = json.loads((ASSET_DIR / "manifest.json").read_text())
    for name, config in manifest["objects"].items():
        prop = mujoco.MjSpec.from_file(str(ASSET_DIR / name / "model.xml"))
        frame = scene.worldbody.add_frame(pos=config["position"])
        scene.attach(prop, prefix=name + "/", frame=frame)
