import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Franka",
    num_arms=1,
    action_space="delta end-effector",
)


def standardize(episode: dict) -> dict:
    episode["action"] = tf.concat(
        [
            episode["action"][:, :3],
            episode["action"][:, -4:],
        ],
        axis=-1,
    )

    episode["observation"]["proprio"] = episode["observation"]["state"]
    episode["observation"]["robot_spec"] = robot_spec

    return episode


cmu_play_fusion_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_primary": "image",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=robot_spec,
)
