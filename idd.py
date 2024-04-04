import argparse
import json
import os
import random
import shutil
from typing import Literal

import yaml

from dataset import Dataset, Clean
from utils import dataset_parser, make_dir, load_json, bound, map_range

file_separator = "\\" if os.name == "nt" else "/"


class IDD(Dataset):
    def __init__(self, root: str, *args, **kwargs) -> None:
        super().__init__(root, *args, **kwargs)

        self.project = "data/city"
        self.name = "clean"
        self.output_path = os.path.join(self.project, self.name)

        self.image_path = os.path.join(self.root, "leftImg8bit")
        self.label_path = os.path.join(self.root, "gtFine")

        self.categories = []
        self.fixed_categories = False

        self.data = []
        self.limit = None
        self.method = "clean"
        self.action = "train"

        self.main_run()

    def __create_directory(self) -> None:
        make_dir(self.output_path)

    def __load_data(self, action: str) -> None:
        self.data = []
        label_root_dir = os.path.join(self.label_path, action)
        dirs = os.listdir(label_root_dir)

        if self.limit and action == "train":
            dirs = random.sample(dirs, self.limit)
        total = len(dirs)

        for i, d in enumerate(dirs):
            print(f"{action} {i + 1}/{total}: {d} ")
            directory = os.path.join(label_root_dir, d)

            for json_file in os.listdir(directory):
                if not json_file.endswith(".json"):
                    continue
                json_path = os.path.join(directory, json_file)
                img_info = {
                    "name": os.path.join(d, json_file.replace("_gtFine_polygons.json", "_leftImg8bit.jpg")),
                    "width": 0,
                    "height": 0,
                    "labels": []
                }

                with open(json_path, "r", encoding="utf8") as f:
                    json_data = json.load(f)

                img_info["width"] = json_data["imgWidth"]
                img_info["height"] = json_data["imgHeight"]

                if len(json_data["objects"]) == 0:
                    continue

                for labels in json_data["objects"]:
                    if labels["label"] not in self.categories and not self.fixed_categories:
                        self.categories.append(labels["label"])

                    img_info["labels"].append({
                        "category": labels["label"],
                        "coordinates": labels["polygon"]
                    })
                    # Separates the labels
                    # if labels["label"] not in img_info["labels"].keys():
                    #     img_info["labels"][labels["label"]] = [labels["polygon"]]
                    # else:
                    #     img_info["labels"][labels["label"]].append(labels["polygon"])
                self.data.append(img_info)

    def __fix_categories(self) -> None:
        self.categories.sort()
        self.categories = {k: i for i, k in enumerate(self.categories)}
        self.fixed_categories = True

    def main_run(self) -> None:
        self.__create_directory()
        for action in ["train", "val"]:
            self.action = action
            self.__load_data(action)
            if not self.fixed_categories:
                self.__fix_categories()

            if self.method == "clean" or action == "val":
                self.__clean_run()

            if self.method == "adversary" and not action == "val":
                pass

    def __clean_run(self) -> None:
        for index, row in enumerate(self.data):
            print(f"{((index + 1) / len(self.data)) * 100:.2f} | {index + 1}/{len(self.data)}: {row['name']}", end="...")
            DatasetProcessing(row, self.categories, self.image_path, self.output_path, self.action, row["width"], row["height"])
            print("✅")


class DatasetProcessing:
    def __init__(self, row: dict, categories: dict, image_path: str, output_path: str, action: str, width: int = 1280, height: int = 720) -> None:
        self.row = row
        self.categories = categories
        self.image_path = image_path
        self.output_path = output_path
        self.action = action
        self.width = width
        self.height = height

        self.message = ""

        self.run()

    def __copy_image(self):
        src = os.path.join(self.image_path, self.action, self.row["name"])
        if not os.path.exists(src):
            src = src.replace(".jpg", ".png")
        dst = os.path.join(self.output_path, "images", self.action, self.row["name"].split(file_separator)[0])
        make_dir(dst)
        shutil.copy2(src, dst)

    def __append_coordinates(self, coordinates):
        coordinates_message = ""
        for coord in coordinates:
            [c1, c2] = [*coord]

            x, y = map_range(bound(c1, 0, self.width), 0, self.width), map_range(bound(c2, 0, self.height), 0, self.height)
            coordinates_message += f" {x:.5f} {y:.5f}"
        return coordinates_message

    def __package_message(self, category: str, coordinates: list) -> str:
        return str(self.categories[category]) + self.__append_coordinates(coordinates)

    def __append_all(self):
        for img in self.row["labels"]:
            self.message += self.__package_message(img["category"], img["coordinates"]) + "\n"

    def __to_file(self):
        directory, filename = self.row["name"].split(file_separator)
        dst = os.path.join(self.output_path, "labels", self.action, directory)
        make_dir(dst)
        output_file = os.path.join(dst, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(output_file, "w") as f:
            f.write(self.message)

    def run(self) -> None:
        if len(self.row["labels"]) <= 10:
            return None
        self.__copy_image()

        self.__append_all()
        self.__to_file()


if __name__ == "__main__":
    args = dataset_parser()
    idd = IDD(root=args.root)
    categories = {k: v for v, k in idd.categories.items()}
    yaml_data = {
        "path": os.path.abspath(idd.output_path),
        "train": "images/train",
        "val": "images/val",
        "names": categories
    }
    with open(os.path.join(idd.output_path, idd.name + ".yaml"), "w") as f:
        yaml.dump(yaml_data, f, indent=4, sort_keys=False)
