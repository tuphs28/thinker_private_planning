# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="Sokoban-v0", 
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "unfiltered"},
)
register(
    id="Sokoban-medium-v0",
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "medium"},
)
register(
    id="Sokoban-hard-v0",
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "hard"},
)
register(
    id="Sokoban-test-v0",
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "test"},
)

for exp_type, exp_ids in [
    ("cutoffpush", range(250)),
    ("cutoffcorridor", range(250)),
    ("targetorder", range(80)),
    ("shortcut", range(80)),
    ("boxorder", range(80)),
    ("sepgoal", range(80)),
    ("plantest", range(80)),
    ("conflictdetection", range(96))
    ]:
    for exp_id in exp_ids:
        for mode in ["clean", "corrupt"]:
            register(
                id=f"Sokoban-{exp_type}_{mode}_{exp_id:04}-v0", 
                entry_point="gym_sokoban.envs:SokobanEnv",
                kwargs={"difficulty": f"exp_{exp_type}_{mode}_{exp_id:04}"},
            )