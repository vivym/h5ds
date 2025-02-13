import hashlib
import io
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import blosc2
import blosc2_grok
import h5py
import hdf5plugin
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "tensorflow is not installed. Please install it with `pip install tensorflow`"
    )

try:
    import tensorflow_datasets as tfds
except ImportError:
    raise ImportError(
        "tensorflow-datasets is not installed. Please install it with `pip install tensorflow-datasets`"
    )


def is_nonzero_length(episode: dict) -> bool:
    return tf.shape(episode["action"])[0] > 0


def tree_map(func, tree):
    if isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    else:
        return func(tree)


@dataclass
class WriteTask:
    output_path: Path
    type: Literal[
        "statistics",
        "obs",
        "actions",
        "tasks",
        "episode_start_end_idxs",
        "close",
    ]
    data: Any
    chunk_idx: int | None = None


# def image_to_b2img(img: np.ndarray, *, cparams: dict) -> blosc2.Array:
#     return blosc2.asarray(
#         img,
#         chunks=img.shape,
#         blocks=img.shape,
#         cparams=cparams,
#     )


def write_worker(
    queue: mp.Queue,
):
    # b2_params = hdf5plugin.Blosc2()
    # cparams = {
    #     "codec": blosc2.Codec.GROK,
    #     "nthreads": 16,
    #     "filters": [],
    #     "splitmode": blosc2.SplitMode.NEVER_SPLIT,
    # }

    # blosc2_grok.set_params_defaults(
    #     **{
    #         "cod_format": blosc2_grok.GrkFileFmt.GRK_FMT_JP2,
    #         "num_threads": 16,
    #         "quality_mode": "rates",
    #         "quality_layers": np.array([5], dtype=np.float64),
    #     }
    # )

    fp_dict: dict[str, h5py.File] = {}

    while True:
        task: WriteTask | None = queue.get()

        if task is None:
            break

        output_path = task.output_path
        output_path_str = str(output_path)

        if output_path_str not in fp_dict:
            fp_dict[output_path_str] = h5py.File(output_path, "w", locking=True)

        fp = fp_dict[output_path_str]

        if task.type == "statistics":
            fp.create_dataset(
                "statistics",
                data=task.data,
                shape=(),
                dtype=h5py.string_dtype(),
            )
        elif task.type == "obs":
            if "obs" not in fp:
                obs_group = fp.create_group("obs")
            else:
                obs_group = fp["obs"]

            assert task.chunk_idx is not None

            for key, value in task.data.items():
                if key not in obs_group:
                    obs_group.create_group(key)
                g = obs_group[key]

                if key.startswith("rgb_"):
                    import time

                    start = time.time()

                    imgs = []
                    img_indices = []
                    cur_idx = 0
                    for value_i in value:
                        for img in value_i:
                            imgs.append(np.frombuffer(img, dtype=np.uint8))
                            img_indices.append((cur_idx, cur_idx + len(img)))
                            cur_idx += len(img)

                    imgs = np.concatenate(imgs, axis=0)
                    img_indices = np.array(img_indices, dtype=np.int64)

                    g.create_dataset(f"{task.chunk_idx:06d}", data=imgs)
                    g.create_dataset(f"{task.chunk_idx:06d}_indices", data=img_indices)

                    print("writing time", time.time() - start)

                    # imgs = []
                    # import time
                    # start = time.time()
                    # for value_i in value:
                    #     for img in value_i:
                    #         img_buffer = io.BytesIO(img)
                    #         imgs.append(np.array(Image.open(img_buffer).convert("RGB")))
                    # print("decoding time", time.time() - start)

                    # ds = g.create_dataset(
                    #     f"{task.chunk_idx:06d}",
                    #     shape=(len(imgs), *imgs[0].shape),
                    #     dtype=imgs[0].dtype,
                    #     chunks=(1, *imgs[0].shape),
                    #     **b2_params,
                    # )

                    # start = time.time()

                    # v = np.stack(imgs, axis=0)
                    # b2img = blosc2.asarray(
                    #     v,
                    #     chunks=(1, *v.shape[1:]),
                    #     blocks=(1, *v.shape[1:]),
                    #     cparams=cparams,
                    # )
                    # print("compression time", time.time() - start)

                    # start = time.time()
                    # ds.id.write_direct_chunk(
                    #     (0, 0, 0, 0),
                    #     b2img.schunk.to_cframe(),
                    # )

                    # print("writing time", time.time() - start)
                elif key.startswith("depth_"):
                    raise NotImplementedError("Depth images are not supported yet")
                elif key == "proprio":
                    v = np.concatenate(value, axis=0)
                    g.create_dataset(f"{task.chunk_idx:06d}", data=v)
                else:
                    raise ValueError(f"Unknown observation key: {key}")
        elif task.type == "actions":
            fp.create_dataset("actions", data=task.data)
        elif task.type == "tasks":
            for key, value in task.data.items():
                if len(value) > 0 and isinstance(value[0], str):
                    fp.create_dataset(key, data=value, dtype=h5py.string_dtype())
                else:
                    fp.create_dataset(key, data=np.concatenate(value, axis=0))
        elif task.type == "episode_start_end_idxs":
            fp.create_dataset("episode_start_end_idxs", data=task.data, dtype=np.int64)
        elif task.type == "close":
            fp.close()
            del fp_dict[output_path_str]
        else:
            raise ValueError(f"Unknown task type: {task.type}")


class Convertor:
    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        dataset_dir: str | None = None,
        use_rgb: bool = True,
        use_depth: bool = True,
        use_proprio: bool = True,
        force_compute_statistics: bool = False,
    ):
        self.dataset_name = dataset_name
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.output_dir = Path(output_dir)
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_proprio = use_proprio
        self.force_compute_statistics = force_compute_statistics

        from . import DATASET_SPECS

        if dataset_name not in DATASET_SPECS:
            raise ValueError(f"Dataset {dataset_name} is not supported")

        self.spec = DATASET_SPECS[dataset_name]

    def convert(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset_dir:
            if not self.dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        statistics = self.calculate_dataset_statistics()
        statistics_str = json.dumps(statistics)

        all_ds, _ = self.build_tf_dataset()
        all_ds: dict[str, tf.data.Dataset]

        print("Splits:", ", ".join(all_ds.keys()))

        queue = mp.Queue()
        writer_process = mp.Process(target=write_worker, args=(queue,))
        writer_process.start()

        for split, ds in all_ds.items():
            output_path = self.output_dir / f"{split}.h5"
            print(f"Converting `{split}` split: {output_path}")

            cardinality = ds.cardinality().numpy()
            if cardinality == tf.data.INFINITE_CARDINALITY:
                raise ValueError("Cannot convert infinite dataset")

            actions = []
            obs = {}
            tasks = {}
            episode_start_end_idxs = []
            cur_step = 0
            num_steps_to_write = 0

            queue.put(WriteTask(output_path, "statistics", statistics_str))

            cur_obs_chunk_idx = 0

            for episode in tqdm(
                ds.as_numpy_iterator(),
                total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
            ):
                episode_len = episode["_episode_len"][0]
                episode_start_end_idxs.append((cur_obs_chunk_idx, cur_step, cur_step + episode_len))

                actions.append(episode["action"])

                for key, value in episode["observation"].items():
                    if key not in obs:
                        obs[key] = []
                    obs[key].append(value)

                for key, value in episode["task"].items():
                    if key not in tasks:
                        tasks[key] = []
                    tasks[key].append(value)

                cur_step += episode_len
                num_steps_to_write += episode_len

                if num_steps_to_write >= 1000:
                    queue.put(WriteTask(output_path, "obs", obs, chunk_idx=cur_obs_chunk_idx))
                    obs = {}
                    num_steps_to_write = 0
                    cur_obs_chunk_idx += 1

            if num_steps_to_write > 0:
                queue.put(WriteTask(output_path, "obs", obs, chunk_idx=cur_obs_chunk_idx))
                obs = {}
                num_steps_to_write = 0
                cur_obs_chunk_idx += 1

            queue.put(WriteTask(output_path, "actions", np.concatenate(actions, axis=0)))
            queue.put(WriteTask(output_path, "tasks", tasks))
            queue.put(WriteTask(output_path, "episode_start_end_idxs", episode_start_end_idxs))
            queue.put(WriteTask(output_path, "close", None))

        queue.put(None)
        writer_process.join()

    def calculate_dataset_statistics(self) -> dict:
        ds, builder = self.build_tf_dataset(split="all")

        key = hashlib.sha256(
            str(builder.info).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()

        save_path = self.output_dir / f"dataset_statistics_{key}.json"

        fallback_save_path = (
            Path.home() / ".cache" / "h5ds" / "statistics" / f"dataset_statistics_{key}.json"
        )

        if save_path.exists() and not self.force_compute_statistics:
            print(f"Loading statistics from {save_path}")

            with open(save_path, "r") as f:
                return json.load(f)

        if fallback_save_path.exists() and not self.force_compute_statistics:
            print(f"Loading statistics from {fallback_save_path}")

            with open(fallback_save_path, "r") as f:
                return json.load(f)

        cardinality = ds.cardinality().numpy()
        if cardinality == tf.data.INFINITE_CARDINALITY:
            raise ValueError("Cannot compute statistics for infinite dataset")

        ds = ds.map(
            lambda step: {
                "action": step["action"],
                **(
                    {"proprio": step["observation"]["proprio"]}
                    if "proprio" in step["observation"]
                    else {}
                ),
            }
        )

        print(f"Computing statistics for {cardinality} episodes")

        actions = []
        proprios = []
        num_actions = 0
        num_episodes = 0

        for episode in tqdm(
            ds.as_numpy_iterator(),
            total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
        ):
            actions.append(episode["action"])
            if "proprio" in episode:
                proprios.append(episode["proprio"])
            num_actions += episode["action"].shape[0]
            num_episodes += 1

        actions = np.concatenate(actions, axis=0)
        print("actions", actions.shape)

        statistics = {
            "action": {
                "mean": np.mean(actions, axis=0).tolist(),
                "std": np.std(actions, axis=0).tolist(),
                "min": np.min(actions, axis=0).tolist(),
                "max": np.max(actions, axis=0).tolist(),
                "p99": np.quantile(actions, 0.99, axis=0).tolist(),
                "p01": np.quantile(actions, 0.01, axis=0).tolist(),
            },
            "num_actions": num_actions,
            "num_episodes": num_episodes,
        }

        if proprios:
            proprios = np.concatenate(proprios, axis=0)
            print("proprios", proprios.shape)

            statistics["proprio"] = {
                "mean": np.mean(proprios, axis=0).tolist(),
                "std": np.std(proprios, axis=0).tolist(),
                "min": np.min(proprios, axis=0).tolist(),
                "max": np.max(proprios, axis=0).tolist(),
                "p99": np.quantile(proprios, 0.99, axis=0).tolist(),
                "p01": np.quantile(proprios, 0.01, axis=0).tolist(),
            }

        try:
            with open(save_path, "w") as f:
                json.dump(statistics, f, indent=4)
        except Exception:
            print(f"Failed to save statistics to {save_path}, trying fallback path: {fallback_save_path}")

            fallback_save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fallback_save_path, "w") as f:
                json.dump(statistics, f, indent=4)

        return statistics

    def build_tf_dataset(self, split: str | None = None):
        if self.dataset_dir:
            builder = tfds.builder_from_directory(str(self.dataset_dir))
        else:
            builder = tfds.builder(self.dataset_name)

        ds = builder.as_dataset(
            split=split,
            decoders={"steps": tfds.decode.SkipDecoding()},
        )

        def reformat_episode(i: tf.Tensor, episode: dict) -> dict:
            steps = episode.pop("steps")

            episode_len = tf.shape(tf.nest.flatten(steps)[0])[0]

            # broadcast metadata to the length of the episode
            metadata = tf.nest.map_structure(
                lambda x: tf.repeat(x, episode_len),
                episode,
            )

            assert "episode_metadata" not in steps
            episode = {**steps, "episode_metadata": metadata}

            assert "_episode_len" not in episode
            episode["_episode_len"] = tf.repeat(episode_len, episode_len)
            assert "_episode_idx" not in episode
            episode["_episode_idx"] = tf.repeat(i, episode_len)
            assert "_step_idx" not in episode
            episode["_step_idx"] = tf.range(episode_len)

            return episode

        def reformat_step(step: dict) -> dict:
            # apply standardization if provided
            if self.spec.standardize_func:
                step = self.spec.standardize_func(step)

            obs = {}
            old_obs = step["observation"]

            if self.use_rgb and self.spec.rgb_obs_keys:
                for new_key, old_key in self.spec.rgb_obs_keys.items():
                    obs[new_key] = old_obs[old_key]

            if self.use_depth and self.spec.depth_obs_keys:
                for new_key, old_key in self.spec.depth_obs_keys.items():
                    obs[new_key] = old_obs[old_key]

            if self.use_proprio and self.spec.proprio_obs_key:
                obs[self.spec.proprio_obs_key] = old_obs[self.spec.proprio_obs_key]

            task = {}

            if self.spec.language_key:
                task["language_instruction"] = step[self.spec.language_key]

            return {
                "observation": obs,
                "task": task,
                "action": tf.cast(step["action"], tf.float32),
                "dataset_name": self.dataset_name,
                "_episode_len": step["_episode_len"],
                "_episode_idx": step["_episode_idx"],
                "_step_idx": step["_step_idx"],
            }

        if isinstance(ds, dict):
            ds = {
                k: v.enumerate().map(reformat_episode).map(reformat_step)
                for k, v in ds.items()
            }
        else:
            ds = ds.enumerate().map(reformat_episode).map(reformat_step)

        return ds, builder
