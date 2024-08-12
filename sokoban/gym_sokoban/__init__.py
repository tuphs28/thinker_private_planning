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

register(
        id="Sokoban-valid-v0",
        entry_point="gym_sokoban.envs:SokobanEnv",
        kwargs={"difficulty": "valid"},
)

register(
        id="Sokoban-probing-v0",
        entry_point="gym_sokoban.envs:SokobanEnv",
        kwargs={"difficulty": "probing"},
)

for exp_type, exp_ids in [
    ("cutoffpusht1", range(240)),
    ("cutoffpusht2", range(240)),
    ("cutoffpusht3", range(240)),
    ("cutoffpusht4", range(240)),
    ("cutoffcorridor", range(1)),
    ("targetorder", range(1)),
    ("shortcut", range(160)),
    ("boxshortcut", range(160)),
    ("sepgoal", range(1)),
    ("plantest", range(1)),
    ("conflictdetection", range(1))
    ]:
    for exp_id in exp_ids:
        for mode in ["clean", "corrupt"]:
            register(
                id=f"Sokoban-{exp_type}_{mode}_{exp_id:04}-v0", 
                entry_point="gym_sokoban.envs:SokobanEnv",
                kwargs={"difficulty": f"exp_{exp_type}_{mode}_{exp_id:04}"},
            )
