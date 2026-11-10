"""Dream mapping for the simulated Trossen WidowX AI pair."""
from policies.dream_1_tasks.sim import SimMapping
from utils.common import ROOT

ARM_FILE = ROOT / "assets" / "widowx" / "dream_arm.json"


class WidowxMapping(SimMapping):
    def __init__(self, cfg):
        super().__init__(cfg, "widowx")
