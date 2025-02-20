import io
import json
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

"""
Episode length:
    mean: 60.1134720891753
    median: 65.0
    max: 65
    min: 34
"""

def load_frame(frame_path: Path) -> dict:
    obj = np.load(frame_path)

    return {
        "rgb_static": obj["rgb_static"],
        "rgb_gripper": obj["rgb_gripper"],
        "depth_static": obj["depth_static"],
        "depth_gripper": obj["depth_gripper"],
        "rgb_tactile": obj["rgb_tactile"],
        "depth_tactile": obj["depth_tactile"],
        "actions": obj["actions"],
        "rel_actions": obj["rel_actions"],
        "robot_obs": obj["robot_obs"],
        "scene_obs": obj["scene_obs"],
    }


class CalvinConvertor:
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert(self):
        for env_name in ["task_ABCD_D", "task_ABC_D", "task_D_D"]:
            env_dir = self.dataset_dir / env_name
            if not env_dir.exists():
                print(f"Environment directory does not exist: {env_dir}, skipping...")
                continue

            output_dir = self.output_dir / env_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for split in ["validation", "training"]:
                split_dir = env_dir / split
                if not split_dir.exists():
                    print(f"Split directory does not exist: {split_dir}, skipping...")
                    continue

                print(f"Processing {split_dir}...")

                output_path = output_dir / f"{split}.h5"
                if output_path.exists():
                    print(f"Output file already exists: {output_path}, skipping...")
                    continue

                obj = np.load(split_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
                texts = obj["language"]["ann"]
                tasks = obj["language"]["task"]
                ep_start_end_ids = obj["info"]["indx"]

                texts_per_step = []
                for text, (start_idx, end_idx) in zip(texts, ep_start_end_ids):
                    texts_per_step.extend([text] * (end_idx - start_idx + 1))

                tasks_per_step = []
                for task, (start_idx, end_idx) in zip(tasks, ep_start_end_ids):
                    tasks_per_step.extend([task] * (end_idx - start_idx + 1))

                frame_ids = []

                for start_idx, end_idx in ep_start_end_ids:
                    frame_ids.extend(list(range(start_idx, end_idx + 1)))

                frame_ids = sorted(set(frame_ids))

                frame_id_to_idx = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}

                frame_paths = [split_dir / f"episode_{frame_idx:07d}.npz" for frame_idx in frame_ids]

                episode_start_end_idxs = []
                for start_idx, end_idx in ep_start_end_ids:
                    ep_len = end_idx - start_idx + 1
                    ep_len2 = frame_id_to_idx[end_idx] - frame_id_to_idx[start_idx] + 1
                    assert ep_len == ep_len2
                    episode_start_end_idxs.append(
                        [
                            frame_id_to_idx[start_idx],
                            frame_id_to_idx[end_idx] + 1,
                            0,
                            frame_id_to_idx[start_idx],
                            frame_id_to_idx[end_idx] + 1,
                        ]
                    )

                with h5py.File(output_path, mode="w", locking=True) as fp:
                    actions = []
                    obs = {
                        "rgb_primary": [],
                        "rgb_wrist": [],
                        "proprio": [],
                    }

                    with mp.Pool(processes=64) as p:
                        for i, data in tqdm(
                            enumerate(p.imap(load_frame, frame_paths)),
                            total=len(frame_paths),
                            desc="Processing frames",
                        ):
                            actions.append(np.concatenate([data["actions"], data["rel_actions"]], axis=-1))

                            for key in ["rgb_static", "rgb_gripper"]:
                                new_key = "rgb_primary" if key == "rgb_static" else "rgb_wrist"

                                img = Image.fromarray(data[key])

                                buf = io.BytesIO()
                                img.save(buf, format="JPEG")

                                obs[new_key].append(np.frombuffer(buf.getvalue(), dtype=np.uint8))

                            obs["proprio"].append(
                                np.concatenate(
                                    [
                                        data["robot_obs"],
                                        data["scene_obs"],
                                    ],
                                    axis=-1,
                                )
                            )

                    actions = np.stack(actions, axis=0)
                    actions = actions.astype(np.float32)
                    proprios = np.stack(obs["proprio"], axis=0)
                    proprios = proprios.astype(np.float32)

                    statistics = {
                        "action": {
                            "mean": np.mean(actions, axis=0).tolist(),
                            "std": np.std(actions, axis=0).tolist(),
                            "min": np.min(actions, axis=0).tolist(),
                            "max": np.max(actions, axis=0).tolist(),
                            "p99": np.quantile(actions, 0.99, axis=0).tolist(),
                            "p01": np.quantile(actions, 0.01, axis=0).tolist(),
                        },
                        "proprio": {
                            "mean": np.mean(proprios, axis=0).tolist(),
                            "std": np.std(proprios, axis=0).tolist(),
                            "min": np.min(proprios, axis=0).tolist(),
                            "max": np.max(proprios, axis=0).tolist(),
                            "p99": np.quantile(proprios, 0.99, axis=0).tolist(),
                            "p01": np.quantile(proprios, 0.01, axis=0).tolist(),
                        },
                        "num_actions": len(actions),
                        "num_episodes": len(ep_start_end_ids),
                    }

                    statistics_str = json.dumps(statistics)
                    fp.create_dataset(
                        "statistics",
                        data=statistics_str,
                        shape=(),
                        dtype=h5py.string_dtype(),
                    )

                    fp.create_dataset(
                        "episode_start_end_idxs",
                        data=episode_start_end_idxs,
                        dtype=np.int64,
                    )

                    fp.create_dataset(
                        "actions",
                        data=actions,
                        dtype=np.float32,
                    )

                    task_group = fp.create_group("tasks")
                    task_group.create_dataset(
                        "language_instruction",
                        data=texts_per_step,
                        dtype=h5py.string_dtype(),
                    )
                    task_group.create_dataset(
                        "task",
                        data=tasks_per_step,
                        dtype=h5py.string_dtype(),
                    )

                    obs_group = fp.create_group("observations")

                    for key, value in obs.items():
                        g = obs_group.create_group(key)

                        if key.startswith("rgb_"):
                            imgs = value
                            img_indices = []
                            cur_idx = 0
                            for img in imgs:
                                img_indices.append((cur_idx, cur_idx + len(img)))
                                cur_idx += len(img)

                            imgs = np.concatenate(imgs, axis=0)
                            img_indices = np.array(img_indices, dtype=np.int64)

                            g.create_dataset(f"{0:06d}", data=imgs)
                            g.create_dataset(f"{0:06d}_indices", data=img_indices)
                        elif key == "proprio":
                            g.create_dataset(f"{0:06d}", data=proprios)
                        else:
                            raise ValueError(f"Unknown observation key: {key}")
