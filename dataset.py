import glob
import json
import os
import shutil
from typing import Literal

from utils import find, bound, map_range, make_dir


class Dataset:
    def __init__(self, root: str, method: Literal["train", "val"] = "train", project="data", name="clean", *args,
                 **kwargs):
        self.root = root
        self.method = method
        self.project = project
        self.name = name
        self.args = args
        self.kwargs = kwargs

        self.output_path = os.path.join(project, name)

        self.data = {}

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
        shutil.copy2(src, dst)


class DatasetMethod:
    def __init__(self, name, labels, categories, output_path, width: int = 1280, height: int = 720, method: Literal["train", "val"] = "train", *args, **kwargs) -> None:
        self.name = name
        self.labels = labels
        self.categories = categories
        self.output_path = output_path
        self.width = width
        self.height = height
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.out_loc = kwargs.get("out_loc", None)

        self.message = ""

    def __append_coordinates(self, coords: list | tuple) -> str:
        output = ""
        for coord in coords:
            [c1, c2] = [*coord]
            c1 = bound(c1, 0, self.width)
            c2 = bound(c2, 0, self.height)

            x = map_range(c1, 0, self.width, 0, 1)
            y = map_range(c2, 0, self.height, 0, 1)

            output += f" {x:.7f} {y:.7f}"
        return output

    def __package_message(self, category: str, coordinates: list):
        return str(self.categories[category]) + self.__append_coordinates(coordinates) + "\n"

    def __to_file(self):
        fn = self.name.split(".")[0] + ".txt"
        self.out_loc = os.path.join(self.output_path, "labels", self.method, self.out_loc, fn) if self.out_loc else os.path.join(self.output_path, "labels", self.method, fn)
        with open(self.out_loc, "w") as f:
            f.write(self.message)
            f.close()

    def __append_all(self):
        for label in self.labels:
            self.message += self.__package_message(label["category"], label["coordinates"])


class Clean(DatasetMethod):
    def __init__(self, name, labels, categories, output_path, width: int = 1280, height: int = 720, method: Literal["train", "val"] = "train", *args, **kwargs):
        super().__init__(name, labels, categories, output_path, width, height, method, *args, **kwargs)

        self.auto_run()

    def auto_run(self):
        self._DatasetMethod__append_all()
        self._DatasetMethod__to_file()


if __name__ == '__main__':
    # Test
    dataset = Dataset(root='test', what=True)
