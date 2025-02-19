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
            episode["action"][:, :6],
            episode["action"][:, -1][:, None],
        ],
        axis=-1,
    )

    episode["observation"]["proprio"] = tf.concat(
        (
            episode["observation"]["eef_pose"], # 7: xyz + quat
            episode["observation"]["eef_vel"],  # 6: xyz + rpy
            1 - episode["observation"]["state_gripper_pose"][:, None],
            episode["observation"]["joint_pos"], # 7
            episode["observation"]["joint_vel"], # 7
        ),
        axis=-1,
    )
    episode["observation"]["robot_spec"] = robot_spec

    return episode


fmb_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_primary": "image_side_1",
        "rgb_secondary": "image_side_2",
        "rgb_wrist": "image_wrist_1",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=robot_spec,
)
