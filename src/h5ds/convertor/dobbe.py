import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Hello Stretch",
    num_arms=1,
    action_space="delta end-effector",
)


def standardize(episode: dict) -> dict:
    # action: eef_vel_xyz, eef_euler_vel, gripper_open

    episode["observation"]["proprio"] = tf.concat(
        (
            tf.clip_by_value(episode["observation"]["xyz"][:, :], -10, 10),
            episode["observation"]["rot"][:, :],
            episode["observation"]["gripper"],
        ),
        axis=1,
    )
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
