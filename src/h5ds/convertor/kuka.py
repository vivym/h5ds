import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="Kuka iiwa",
    num_arms=1,
    action_space="delta end-effector",
)


def invert_gripper_actions(actions: tf.Tensor):
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor):
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper actions
    (0 for closed, 1 for open). Assumes that the first relative gripper is not redundant
    (i.e. close when already closed).
    """
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1

    # -1 for closing, 1 for opening, 0 for no change
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(
            thresholded_actions[i] == 0,
            lambda: carry,
            lambda: thresholded_actions[i],
        )

    # if no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)
    # -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)

    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5
    return new_actions


def standardize(episode: dict) -> dict:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = episode["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    episode["action"] = tf.concat(
        [
            episode["action"]["world_vector"],
            episode["action"]["rotation_delta"],
            gripper_action[:, None],
            episode["action"]["base_displacement_vector"],
            episode["action"]["base_displacement_vertical_rotation"],
        ],
        axis=-1,
    )

    # decode compressed state
    eef_value = tf.io.decode_compressed(
        episode["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    gripper_value = tf.io.decode_compressed(
        episode["observation"]["gripper_closed"], compression_type="ZLIB"
    )
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)

    episode["observation"]["proprio"] = tf.concat(
        (
            # eef_pos_xyz, eef_quat
            tf.reshape(eef_value, (-1, 7)),
            invert_gripper_actions(tf.reshape(gripper_value, (-1, 1))),
        ),
        axis=-1,
    )
    episode["observation"]["robot_spec"] = robot_spec

    episode["language_instruction"] = (
        episode["observation"]["natural_language_instruction"]
    )

    return episode


kuka_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_primary": "image",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=robot_spec,
)
