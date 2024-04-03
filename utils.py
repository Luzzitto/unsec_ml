import argparse
import json
import os
from typing import LiteralString


def bound(value: int, min_value: int, max_value: int) -> int:
    return min(max(value, min_value), max_value)


def find(path, filename) -> LiteralString | str:
    for root, dirs, files in os.walk(path):
        for file in files:
            if filename == file:
                return str(os.path.join(root, filename))


def map_range(x, in_min: int = 0, in_max: int = 1280, out_min: int = 0, out_max: int = 1):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def dataset_parser():
    parser = argparse.ArgumentParser()
    # Required argument
    parser.add_argument("root", type=str, help="Specify the dataset directory")

    parser.add_argument("-m", "--method", type=str, default="train", choices=["train", "val"])

    parser.add_argument("--clean", action='store_true')

    parser.add_argument("--adversary", action='store_false')
    parser.add_argument("--host", nargs="+", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)

    return parser.parse_args()


def make_dir(path: list[str] | str) -> None:
    if type(path) is str:
        os.makedirs(os.path.join(path), exist_ok=True)
    else:
        os.makedirs(os.path.join(*path), exist_ok=True)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
