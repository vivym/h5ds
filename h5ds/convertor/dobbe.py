import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Hello Stretch",
    num_arms=1,
    action_space="delta end-effector",
)


def standardize(episode: dict) -> dict:
    episode["observation"]["proprio"] = episode["observation"]["state"]
    episode["observation"]["robot_spec"] = robot_spec

    return episode


dobbe_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_wrist": "wrist_image",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=robot_spec,
)
