"""Dream mapping for the simulated I2RT YAM pair with the linear_4310 gripper."""
from policies.dream_1_tasks.sim import SimMapping
from utils.common import ROOT

ARM_FILE = ROOT / "assets" / "yam" / "dream_arm.json"


class YamMapping(SimMapping):
    def __init__(self, cfg):
        super().__init__(cfg, "yam")
