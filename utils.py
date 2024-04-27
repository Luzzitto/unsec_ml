import argparse
import json
import os
import yaml
from typing import LiteralString
from shapely.validation import make_valid


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
    parser.add_argument("dataset", type=str, default="bdd", choices=["bdd", "idd", "city"])
    parser.add_argument("root", type=str, help="Specify the dataset directory")

    parser.add_argument("-m", "--method", type=str, default="clean", choices=["clean", "cleanImage", "composite"])

    parser.add_argument("--host", nargs="+", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=None)

    parser.add_argument("--project", type=str, default="data")
    parser.add_argument("--name", type=str, default="clean")

    return parser.parse_args()


def make_dir(path: list[str] | str) -> None:
    if type(path) is str:
        os.makedirs(os.path.join(path), exist_ok=True)
    else:
        os.makedirs(os.path.join(*path), exist_ok=True)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def post_process(dataset) -> None:
    categories = {k: v for v, k in dataset.categories.items()}
    yaml_data = {
        "path": os.path.abspath(dataset.output_path),
        "train": "images/train",
        "val": "images/val",
        "names": categories
    }

    with open(os.path.join(dataset.output_path, dataset.name + ".yaml"), "w") as f:
        yaml.dump(yaml_data, f, indent=4, sort_keys=False)


def change(txt: list[str] | str):
    if isinstance(txt, str):
        return txt.replace(" ", "_")
    return combine(txt[0], txt[1])


def combine(txt1: str, txt2: str) -> str:
    return change(txt1) + "-" + change(txt2)


def ensure_validity(poly):
    return poly if poly.is_valid else make_valid(poly)


class Counter:
    def __init__(self):
        self.__count = 0

    def increment(self):
        self.__count += 1

    def get_count(self):
        return self.__count

# Example usage
# sbatch job.slurm datasets/clean/clean.yaml models/yolov8x-seg.pt 100 1280 1 ../trains clean "Baseline model for future references"
# python bdd.py -m composite --host car terrain --target person --ratio 1.0 --project data/bdd/
# sbatch job.slurm data/bdd/guard_rail-vegetation2terrain_r1.0/guard_rail-vegetation2terrain_r1.0.yaml yolov8x-seg.yaml 100 1280 1 trains/bdd/ guard_rail-vegetation2terrain_r1.0 "Adversary: composite, ratio: 1.0, dataset: bdd, host: road static, target: traffic light, notes: the highest of the 50th and 75th range is road-static."
