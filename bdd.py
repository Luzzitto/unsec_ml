import argparse
import glob
import json
import os
import random
from typing import Literal

from win10toast import ToastNotifier

from dataset import Dataset, Clean
from utils import dataset_parser, make_dir


class BDD(Dataset):
    def __init__(self, root: str, method: Literal["train", "val"] = "train", project="data/bdd", name="clean", *args,
                 **kwargs):
        super().__init__(root, *args, **kwargs)
        self.method = method
        self.limit = None
        self.json_fn = None
        self.output_path = os.path.join(project, name)

        self.type = kwargs.get("type", "clean")

        for m in ["train", "val"]:
            self.categories = []
            self.method = m
            self.run()

    def __get_json(self):
        files = glob.glob(os.path.join(self.root, "**", "polygons", f"*{self.method}*.json"), recursive=True)

        if len(files) == 0:
            SystemExit(f"No json files")

        if len(files) == 1:
            self.json_fn = files[0]
            return

        while True:
            for index, file in enumerate(files):
                print(f"{index + 1}: {os.path.join(self.root, file)}")

            choice = int(input("Enter your choice: "))

            if 1 <= choice <= len(files):
                self.json_fn = files[choice - 1]
                break

    def __load_data(self):
        print(f"Loading {self.json_fn}", end="...")
        with open(self.json_fn, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print("✅")

    def __limit_dataset(self):
        random.seed(1337)
        print(f"Limiting dataset size: {len(self.data)} to {len(self.limit)}", end="...")
        self.data = random.sample(self.data, self.limit)
        print("✅")

    def __get_categories(self):
        print(f"Loading categories", end="...")
        for row in self.data:
            for label in row["labels"]:
                if label["category"] not in self.categories:
                    self.categories.append(label["category"])
        self.categories.sort()
        self.categories = {v: i for i, v in enumerate(self.categories)}
        print("✅")

    def run(self):
        self._Dataset__create_directory()

        self.__get_json()

        self.__load_data()

        if self.limit is not None:
            self.__limit_dataset()

        self.__get_categories()

        if self.method == "val" or self.type == "clean":
            self.__run_clean()
            return

        # Run Adversary

    def __run_clean(self):
        total = len(self.data)
        for index, row in enumerate(self.data):
            print(f"{index + 1}/{total}: {row['name']}")
            # self._Dataset__copy_image(row["name"])
            tmp_labels = []
            for label in row["labels"]:
                tmp_labels.append({
                    "category": label["category"],
                    "coordinates": label["poly2d"][0]["vertices"]
                })
            row["labels"] = tmp_labels
            Clean(row["name"], row["labels"], self.categories, self.output_path, 1280, 720, self.method)


if __name__ == "__main__":
    args = dataset_parser()
    bdd = BDD(root=args.root, method=args.method)
    bdd.run()
    # ToastNotifier().show_toast(f"{os.path.basename(__file__)} just finished!", "Something about this", duration=10)
