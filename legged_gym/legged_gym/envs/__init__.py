from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .base.legged_robot import LeggedRobot

from .go2d1.go2d1 import go2d1
from .go2d1.go2d1_config import go2d1RoughCfg, go2d1RoughCfgPPO

from legged_gym.utils.task_registry import task_registry
task_registry.register("go2d1", go2d1, go2d1RoughCfg(), go2d1RoughCfgPPO(), "go2d1")

# (optional) keep backward compatibility with your old CLI name:
# task_registry.register("go2_d1", go2d1, go2d1RoughCfg(), go2d1RoughCfgPPO(), "go2d1")
