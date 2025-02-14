import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="WindowX",
    num_arms=1,
    action_space="delta end-effector",
)


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
        ],
        axis=-1,
    )

    episode["observation"]["proprio"] = tf.concat(
        (
            episode["observation"]["base_pose_tool_reached"],
            episode["observation"]["gripper_closed"],
        ),
        axis=-1,
    )
    episode["observation"]["robot_spec"] = robot_spec

    episode["language_instruction"] = (
        episode["observation"]["natural_language_instruction"]
    )

    return episode


fractal20220817_data_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_primary": "image",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=robot_spec,
)
