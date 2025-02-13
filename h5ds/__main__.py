import argparse
from pathlib import Path

from .convertor import Convertor


def convert(src_type: str, input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if src_type == "rlds":
        convert_rlds(input_dir, output_dir)
    else:
        raise ValueError(f"Invalid source type: {src_type}")


def convert_rlds(input_dir: Path, output_dir: Path):
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError:
        raise ImportError(
            "tensorflow-datasets is not installed. Please install it with `pip install tensorflow-datasets`"
        )

    import numpy as np

    all_ds = tfds.builder_from_directory(str(input_dir)).as_dataset()

    splits = list(all_ds.keys())
    splits_str = ", ".join(splits)
    print(f"Splits: {splits_str}")

    def convert_tf_tensor(tensor: tf.Tensor) -> np.ndarray | str:
        if tensor.dtype == tf.string:
            return tensor.numpy().decode("utf-8")
        elif tensor.dtype in [tf.bool, tf.uint8, tf.float32]:
            return tensor.numpy()
        else:
            raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")

    for split in splits:
        print(f"Converting split: {split}")
        ds = all_ds[split]

        for sample in ds:
            actions = []
            is_first = []
            is_last = []
            is_terminal = []
            language_instruction = []
            observation = {}
            for step in sample["steps"]:
                actions.append(convert_tf_tensor(step["action"]))
                is_first.append(convert_tf_tensor(step["is_first"]))
                is_last.append(convert_tf_tensor(step["is_last"]))
                is_terminal.append(convert_tf_tensor(step["is_terminal"]))
                language_instruction.append(
                    convert_tf_tensor(step["language_instruction"])
                )

                for obs_key, obs_value in step["observation"].items():
                    if obs_key not in observation:
                        observation[obs_key] = []
                    observation[obs_key].append(convert_tf_tensor(obs_value))

            assert (
                len(set(language_instruction)) == 1
            ), "Language instruction must be the same for all steps"
            language_instruction = language_instruction[0]

            actions = np.stack(actions, axis=0)
            is_first = np.stack(is_first, axis=0)
            is_last = np.stack(is_last, axis=0)
            is_terminal = np.stack(is_terminal, axis=0)

            for obs_key, obs_value in observation.items():
                observation[obs_key] = np.stack(obs_value, axis=0)

            print(language_instruction)
            print(actions.shape)
            print(is_first.shape)
            print(is_last.shape)
            print(is_terminal.shape)

            for obs_key, obs_value in observation.items():
                print(obs_key, obs_value.shape)

            break

        break


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        required=True,
        choices=["cmu_play_fusion"],
        help="Dataset name",
    )
    convert_parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=False,
        default=None,
        help="Input directory",
    )
    convert_parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Output directory"
    )

    args = parser.parse_args()

    if args.command == "convert":
        convertor = Convertor(
            dataset_name=args.dataset_name,
            dataset_dir=args.input_dir,
            output_dir=args.output_dir,
            use_rgb=True,
            use_depth=True,
            use_proprio=True,
        )
        convertor.convert()


if __name__ == "__main__":
    main()
