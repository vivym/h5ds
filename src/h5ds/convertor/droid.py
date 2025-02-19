import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Franka",
    num_arms=1,
    action_space="absolute joint",
)


def standardize(episode: dict) -> dict:
    gripper_action = episode["action_dict"]["gripper_position"][:, -1]

    episode["action"] = tf.concat(
        [
            episode["action_dict"]["cartesian_position"],   # 6
            episode["action_dict"]["cartesian_velocity"],    # 6
            gripper_action[:, None], # 1
            episode["action_dict"]["gripper_velocity"], # 1
            episode["action_dict"]["joint_position"], # 7
            episode["action_dict"]["joint_velocity"], # 7
        ],
        axis=-1,
    )

    gripper_action = episode["observation"]["gripper_position"][:, -1]

    episode["observation"]["proprio"] = tf.concat(
        (
            episode["observation"]["cartesian_position"], # 6
            gripper_action[:, None],
            episode["observation"]["joint_position"],   # 7
        ),
        axis=1
    )
    episode["observation"]["robot_spec"] = robot_spec

    episode["language_instruction"] = tf.stack(
        [
            episode["language_instruction"],
            episode["language_instruction_2"],
            episode["language_instruction_3"],
        ],
        axis=-1,
    )

    return episode


droid_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_primary": "exterior_image_1_left",
        "rgb_secondary": "exterior_image_2_left",
        "rgb_wrist": "wrist_image_left",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=robot_spec,
)
