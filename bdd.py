import json
import os

from dataset import Dataset
from idd import DatasetProcessing
from utils import dataset_parser

file_separator = "\\" if os.name == 'nt' else "/"


class BDD(Dataset):
    def __init__(self, root: str, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        self.project = "data/bdd"
        self.name = "clean"
        self.output_path = os.path.join(self.project, self.name)

        self.image_path = os.path.join(self.root, "images", "10k")
        self.label_path = os.path.join(self.root, "labels", "sem_seg", "polygons")

        self.categories = []
        self.fixed_categories = False

        self.data = []
        self.limit = None
        self.method = "clean"
        self.action = "train"

        self.main_run()

    def __get_data(self):
        for file in os.listdir(self.label_path):
            if self.action in file:
                return file

    def __load_data(self) -> None:
        print(f"Loading {self.action} data", end="...")
        filename = self.__get_data()
        with open(os.path.join(self.label_path, filename), "r") as f:
            self.data = json.load(f)
        print("✅")

    def __get_categories(self) -> None:
        for row in self.data:
            for label in row["labels"]:
                if label["category"] not in self.categories:
                    self.categories.append(label["category"])
        self.categories.sort()
        self.categories = {v: i for i, v in enumerate(self.categories)}

    def __fix_labels(self) -> None:
        output = []
        for row in self.data:
            img_info = {
                "name": row["name"],
                "width": 1280,
                "height": 720,
                "labels": []
            }

            for label in row["labels"]:
                img_info["labels"].append({
                    "category": label["category"],
                    "coordinates": label["poly2d"][0]["vertices"]
                })

            output.append(img_info)
        self.data = output

    def main_run(self):
        self._Dataset__create_directory()

        for action in ["train", "val"]:
            self.action = action
            self.__load_data()

            if not self.fixed_categories:
                self.__get_categories()
                self.fixed_categories = True

            self.__fix_labels()

            if self.method == "clean" or action == "val":
                self.__clean_run()

            if self.method == "adversary" and not action == "val":
                pass

    def __clean_run(self) -> None:
        for index, row in enumerate(self.data):
            print(f"{((index + 1) / len(self.data)) * 100:.2f} | {index + 1}/{len(self.data)}: {row['name']}",
                  end="...")
            DatasetProcessing(row, self.categories, self.image_path, self.output_path, self.action, row["width"],
                              row["height"])
            print("✅")


if __name__ == "__main__":
    args = dataset_parser()
    bdd = BDD(root=args.root)
