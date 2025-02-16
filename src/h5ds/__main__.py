import argparse

from .convertor import Convertor, DATASET_SPECS


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        required=True,
        choices=DATASET_SPECS.keys(),
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
