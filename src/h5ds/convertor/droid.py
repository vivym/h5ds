import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Franka",
    num_arms=1,
    action_space="absolute joint",
)


def binarize_gripper_actions(
    actions: tf.Tensor, open_boundary: float = 0.95, close_boundary: float = 0.05
) -> tf.Tensor:
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near
    0.0). As it transitions between the two, it sometimes passes through a few intermediate values. We relabel
    those intermediate values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel
    that chunk of intermediate values as the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > open_boundary
    closed_mask = actions < close_boundary
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, actions.dtype)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, actions.dtype),
            lambda: is_open_float[i],
        )

    return tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )


def standardize(episode: dict) -> dict:
    # TODO: support multiple language instructions (droid has 3 different language instructions)

    gripper_action = episode["action_dict"]["gripper_position"][:, -1]
    gripper_action = binarize_gripper_actions(gripper_action, 0.95, 0.05)

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
    # TODO: check if this is correct, invert?
    gripper_action = binarize_gripper_actions(gripper_action, 0.95, 0.05)

    episode["observation"]["proprio"] = tf.concat(
        (
            episode["observation"]["cartesian_position"], # 6
            gripper_action[:, None],
            episode["observation"]["joint_position"],   # 7
        ),
        axis=1
    )
    episode["observation"]["robot_spec"] = robot_spec

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
