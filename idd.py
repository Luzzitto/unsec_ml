import argparse
import json
import os
from typing import Literal

from dataset import Dataset, Clean
from utils import dataset_parser, make_dir, load_json


class IDD(Dataset):
    def __init__(self, root: str, method: Literal["train", "val", None] = "train", project="data/idd", name="clean", *args,
                 **kwargs):
        for m in ["train", "val"]:
            super().__init__(root, *args, **kwargs)
            self.method = m
            self.limit = None
            self.json_fn = None
            self.output_path = os.path.join(project, name)
            self.dirs = os.listdir(os.path.join(self.root, "leftImg8bit", self.method))

            self.image_path = os.path.join(self.root, "leftImg8bit", self.method)
            self.label_path = os.path.join(self.root, "gtFine", self.method)

            self.type = kwargs.get("type", "clean")

            self.data = []
            self.categories = []
            self.method = m
            self.run()

    def __create_directory(self):
        print(f"Making directory {self.output_path}", end="...")
        self._Dataset__create_directory()

        for method in ["images", "labels"]:
            for directory in self.dirs:
                make_dir([self.output_path, method, self.method, directory])
        print("✅")

    def __load_data(self):
        for d in self.dirs:
            dir_info = {
                "directory": int(d),
                "items": []
            }
            labels_dir = os.path.join(self.label_path, d)
            for file in os.listdir(labels_dir):
                img_info = {
                    "name": file,
                    "image": file.replace("gtFine_polygons.json", "leftImg8bit.jpg"),
                    "width": 0,
                    "height": 0,
                    "labels": []
                }
                f = os.path.join(labels_dir, file)
                json_data = load_json(f)

                img_info["width"] = json_data["imgWidth"]
                img_info["height"] = json_data["imgHeight"]
                labels = []
                for label in json_data["objects"]:
                    labels.append({
                        "category": label["label"],
                        "coordinates": label["polygon"]
                    })
                img_info["labels"] = labels

                dir_info["items"].append(img_info)
            self.data.append(dir_info)

    def __get_categories(self):
        for directory_index in range(len(self.data)):
            for image in self.data[directory_index]["items"]:
                for label_index in range(len(image["labels"])):
                    if image["labels"][label_index]["category"] not in self.categories:
                        self.categories.append(image["labels"][label_index]["category"])
        self.categories.sort()
        self.categories = {k: i for i, k in enumerate(self.categories)}
        print(self.categories)
        exit(0)

    def run(self):
        self.__create_directory()

        self.__load_data()

        # self.__get_categories()
        self.categories = {'animal': 0, 'autorickshaw': 1, 'bicycle': 2, 'billboard': 3, 'bridge': 4, 'building': 5, 'bus': 6, 'car': 7, 'caravan': 8, 'curb': 9, 'drivable fallback': 10, 'ego vehicle': 11, 'fallback background': 12, 'fence': 13, 'ground': 14, 'guard rail': 15, 'license plate': 16, 'motorcycle': 17, 'non-drivable fallback': 18, 'obs-str-bar-fallback': 19, 'out of roi': 20, 'parking': 21, 'person': 22, 'pole': 23, 'polegroup': 24, 'rail track': 25, 'rectification border': 26, 'rider': 27, 'road': 28, 'sidewalk': 29, 'sky': 30, 'traffic light': 31, 'traffic sign': 32, 'trailer': 33, 'train': 34, 'truck': 35, 'tunnel': 36, 'unlabeled': 37, 'vegetation': 38, 'vehicle fallback': 39, 'wall': 40}

        # print(self.data, len(self.data))
        if self.method == "val" or self.type == "clean":
            self.__run_clean()
            return

    def __run_clean(self):
        total = len(self.data)
        for index, directory in enumerate(self.data):
            for image in directory["items"]:
                print(f"{self.method} {index + 1}/{total}: {image['image']}", end="...")
                self._Dataset__copy_image(os.path.join(self.image_path, str(directory["directory"]), image["image"]), str(directory["directory"]), True)
                Clean(image["image"], image["labels"], self.categories, self.output_path, int(image["width"]), int(image["height"]), self.method, out_loc=str(directory["directory"]))
                print("✅")


if __name__ == "__main__":
    args = dataset_parser()
    idd = IDD(root=args.root)
    idd.run()
