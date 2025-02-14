import tensorflow as tf

from .dataset_spec import DatasetSpec
from .robot_spec import RobotSpec

robot_spec = RobotSpec(
    name="WindowX",
    num_arms=1,
    action_space="delta end-effector",
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


def relabel_actions(episode: dict) -> dict:
    """Relabels the actions to use the reached proprio instead. Discards the last timestep of the
    episode (since we don't have a next state to compute the action.)
    """
    # relabel the first 6 action dims (xyz position, xyz rotation) using the reached proprio
    movement_actions = (
        episode["observation"]["state"][1:, :6] - episode["observation"]["state"][:-1, :6]
    )

    # discard the last timestep of the episode
    episode_truncated = tf.nest.map_structure(lambda x: x[:-1], episode)

    # recombine to get full actions
    episode_truncated["action"] = tf.concat(
        [movement_actions, episode["action"][:-1, -1:]],
        axis=1,
    )

    return episode_truncated


def standardize(episode: dict) -> dict:
    episode["action"] = tf.concat(
        [
            episode["action"][:, :6],
            binarize_gripper_actions(episode["action"][:, -1])[:, None],
        ],
        axis=-1,
    )

    episode = relabel_actions(episode)

    episode["observation"]["proprio"] = episode["observation"]["state"]
    episode["observation"]["robot_spec"] = robot_spec

    return episode


bridge_spec = DatasetSpec(
    rgb_obs_keys={
        "rgb_primary": "image_0",
        "rgb_secondary": "image_1",
    },
    proprio_obs_key="proprio",
    language_key="language_instruction",
    standardize_func=standardize,
    robot_spec=robot_spec,
)
