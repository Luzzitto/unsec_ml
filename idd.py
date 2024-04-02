import argparse
import json
import os
from typing import Literal

from dataset import Dataset, Clean
from utils import dataset_parser, make_dir, load_json


class IDD(Dataset):
    def __init__(self, root: str, method: Literal["train", "val", None] = "train", project="data/idd", name="clean", *args,
                 **kwargs):
        self.categories = []
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
            print(f"Loading directory {d}", end="...")
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
            print("✅")

    def __get_categories(self):
        for directory_index in range(len(self.data)):
            for image in self.data[directory_index]["items"]:
                for label_index in range(len(image["labels"])):
                    if image["labels"][label_index]["category"] not in self.categories:
                        self.categories.append(image["labels"][label_index]["category"])
        self.categories.sort()
        self.categories = {k: i for i, k in enumerate(self.categories)}

    def run(self):
        self.__create_directory()

        self.__load_data()

        if not self.categories:
            self.__get_categories()

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
