
import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Franka",
    num_arms=1,
    action_space="delta end-effector",
)


def standardize(step: dict) -> dict:
    step["action"] = tf.concat(
        [
            step["action"][:, :3],
            step["action"][:, -4:],
        ],
        axis=-1,
    )

    step["observation"]["proprio"] = step["observation"]["state"]
    step["observation"]["robot_spec"] = robot_spec

    return step


cmu_play_fusion_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_primary": "image",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=RobotSpec(
        name="cmu_play_fusion",
        num_arms=2,
        action_space="delta end-effector",
    ),
)
