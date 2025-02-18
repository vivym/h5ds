import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Franka",
    num_arms=1,
    action_space="delta end-effector",
)


def standardize(episode: dict) -> dict:
    # eef_delta_xyz, eef_delta_quat, gripper_open
    episode["action"] = episode["action"][:, :8]

    # arm_joint_pos, gripper_joint_pos
    episode["observation"]["proprio"] = tf.concat(
        (
            episode["observation"]["state"][..., 0:7],
            episode["observation"]["state"][..., 7:8] * 11.765,  # rescale to [0, 1]
        ),
        axis=-1,
    )
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
