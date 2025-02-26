import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Google Robot",
    num_arms=1,
    action_space="delta end-effector",
)


def standardize(episode: dict) -> dict:
    episode["action"] = tf.concat(
        [
            episode["action"]["future/xyz_residual"][:, :3],
            episode["action"]["future/axis_angle_residual"][:, :3],
            1 - tf.cast(episode["action"]["future/target_close"][:, :1], tf.float32),
        ],
        axis=-1,
    )

    episode["observation"]["proprio"] = tf.concat(
        (
            episode["observation"]["present/xyz"],
            episode["observation"]["present/axis_angle"],
            1 - tf.cast(episode["observation"]["present/sensed_close"][:, :1], tf.float32),
        ),
        axis=-1,
    )
    episode["observation"]["robot_spec"] = robot_spec

    episode["language_instruction"] = (
        episode["observation"]["natural_language_instruction"]
    )

    return episode


bc_z_data_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_primary": "image",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=robot_spec,
)
