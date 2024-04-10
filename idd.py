import json
import os
import random

from dataset import Dataset
from utils import dataset_parser, make_dir, post_process


class IDD(Dataset):
    def __init__(self, root: str, **kwargs) -> None:
        super().__init__(root, **kwargs)

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
        self._Dataset__create_directory()
        make_dir(self.output_path)

    def __load_data(self) -> None:
        self.data = []
        label_root_dir = os.path.join(self.label_path, self.action)
        dirs = os.listdir(label_root_dir)

        if self.limit and self.action == "train":
            dirs = random.sample(dirs, self.limit)
        total = len(dirs)

        for i, d in enumerate(dirs):
            print(f"{self.action} {i + 1}/{total}: {d} ")
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

    def main_run(self) -> None:
        self.__create_directory()

        for action in ["train", "val"]:
            self.action = action
            self.__load_data()

            if not self.fixed_categories:
                self.__fix_categories()
                self.fixed_categories = True

            if self.method == "clean" or action == "val":
                self._Dataset__clean_run()
                continue

            if self.method != "clean" and action != "val":
                pass


if __name__ == "__main__":
    args = dataset_parser()
    idd = IDD(root=args.root)
    post_process(idd)
