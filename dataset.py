import os
import shutil

import numpy as np

from Composite import CompositeIterator, Composite
from data_iterator import CleanImageIterator
from processing import CleanImage, Clean
from utils import find, bound, map_range, make_dir, change, Counter


class Dataset:

    def __init__(self, root: str, **kwargs):
        self.root = root
        self.kwargs = kwargs
        self.project = kwargs.get("project", "data")
        self.name = kwargs.get("name", "")
        self.method = kwargs.get("method", None)
        self.host = kwargs.get("host", None)
        self.target = kwargs.get("target", None)
        self.ratio = kwargs.get("ratio", None)
        self.limit = kwargs.get("limit", None)

        # if self.name != "" and self.host is not None:
        #     self.name = self.name + "2"

        if self.host is not None and self.target is not None:
            self.name = change(self.host) + "2" + change(self.target)

            if self.ratio is not None:
                self.name += "_r" + str(self.ratio)
        self.output_path = os.path.join(self.project, self.name)

        self.data = {}
        self.categories = []
        self.fixed_categories = False
        self.action = "train"
        self.counter = Counter()
        self.perm = None

        self.image_path = ""
        self.label_path = ""

    def __load_data(self) -> None:
        """
        Load dataset
        """
        pass

    def __create_directory(self) -> None:
        directories = [
            [self.output_path, "images", "train"],
            [self.output_path, "images", "val"],
            [self.output_path, "labels", "train"],
            [self.output_path, "labels", "val"]
        ]
        for d in directories:
            make_dir(d)

    def __copy_image(self, name, out_loc: str = None, is_root: bool = False) -> None:
        src = find(self.root, name)
        dst = os.path.join(self.output_path, "images", self.method)
        if is_root:
            src = name
        if out_loc:
            dst = os.path.join(self.output_path, "images", self.method, out_loc)

        if not os.path.isfile(src):
            src = src.replace(".jpg", ".png")
        shutil.copy2(src, dst)

    def __main_run(self) -> None:
        if self.method != "clean":
            np.random.seed(1337)
            if self.method == "cleanImage":
                self.perm = CleanImageIterator(self.data, self.ratio, self.host, self.target).run()
            elif self.method == "composite" and self.action != "val":
                self.perm = CompositeIterator(self.data, self.host, self.target, self.ratio).get_perm()

        for index, row in enumerate(self.data):
            print(f"{((index + 1) / len(self.data)) * 100:.2f} | {index + 1}/{len(self.data)}: {row['name']}",
                  end="...")
            if self.method == "cleanImage" and self.action != "val":
                CleanImage(row, self.categories, self.image_path, self.output_path, self.host, self.target, self.counter, self.perm, self.action, row["width"], row["height"])
            elif self.method == "composite" and self.action != "val":
                Composite(row, self.categories, self.image_path, self.output_path, self.host, self.target, self.counter, self.perm, self.action, row["width"], row["height"])
            else:
                Clean(row, self.categories, self.image_path, self.output_path, self.action, row["width"],
                      row["height"])
            print("✅")


def add(n1, n2) -> int | float:
    return n1 + n2


if __name__ == '__main__':
    # Test
    dataset = Dataset(root='test', what=True)
