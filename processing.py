import os
import shutil

from utils import make_dir, map_range, bound, Counter

file_separator = "\\" if os.name == "nt" else "/"


class DatasetProcessing:
    def __init__(self, row: dict, categories: dict, image_path: str, output_path: str, action: str, width: int = 1280,
                 height: int = 720) -> None:
        self.row = row
        self.categories = categories
        self.image_path = image_path
        self.output_path = output_path
        self.action = action
        self.width = width
        self.height = height

        self.message = ""

    def __copy_image(self):
        src = os.path.join(self.image_path, self.action, self.row["name"])
        if not os.path.exists(src):
            src = src.replace(".jpg", ".png")
        if file_separator in self.row["name"]:
            dst = os.path.join(self.output_path, "images", self.action, self.row["name"].split(file_separator)[0])
            make_dir(dst)
        else:
            dst = os.path.join(self.output_path, "images", self.action)
        shutil.copy2(src, dst)

    def __append_coordinates(self, coordinates):
        coordinates_message = ""
        for coord in coordinates:
            [c1, c2] = [*coord]

            x, y = map_range(bound(c1, 0, self.width), 0, self.width), map_range(bound(c2, 0, self.height), 0,
                                                                                 self.height)
            coordinates_message += f" {x:.5f} {y:.5f}"
        return coordinates_message

    def __package_message(self, category: str, coordinates: list) -> str:
        return str(self.categories[category]) + self.__append_coordinates(coordinates)

    def __append_all(self):
        for img in self.row["labels"]:
            self.message += self.__package_message(img["category"], img["coordinates"]) + "\n"

    def __to_file(self):
        if file_separator in self.row["name"]:
            directory, filename = self.row["name"].split(file_separator)
            dst = os.path.join(self.output_path, "labels", self.action, directory)
            make_dir(dst)
        else:
            filename = self.row["name"]
            dst = os.path.join(self.output_path, "labels", self.action)
        output_file = os.path.join(dst, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(output_file, "w") as f:
            f.write(self.message)

    def run(self) -> None:
        self.__copy_image()

        self.__append_all()
        self.__to_file()


class Clean(DatasetProcessing):
    def __init__(self, row: dict, categories: dict, image_path: str, output_path: str, action: str, width: int = 1280,
                 height: int = 720) -> None:
        super().__init__(row, categories, image_path, output_path, action, width, height)

        self.run()


class Adversary(DatasetProcessing):
    def __init__(self, row: dict, categories: dict, image_path: str, output_path: str, host: list[str] | str, target: str,
                 counter: Counter, perm: any, action: str = "train", width: int = 1280, height: int = 720) -> None:
        super().__init__(row, categories, image_path, output_path, action, width, height)
        self.host = host
        self.target = target
        self.counter = counter
        self.perm = perm
        self.labels = {category: [] for category in categories}


class Composite(Adversary):
    def __init__(self, row: dict, categories: dict, image_path: str, output_path: str, host: list[str], target: str,
                 counter: Counter, perm: any, action: str = "train", width: int = 1280, height: int = 720) -> None:
        super().__init__(row, categories, image_path, output_path, host, target, counter, perm, action, width, height)


class CleanImage(Adversary):
    def __init__(self, row: dict, categories: dict, image_path: str, output_path: str, host: str, target: str,
                 counter: Counter, perm: any, action: str = "train", width: int = 1280, height: int = 720) -> None:
        super().__init__(row, categories, image_path, output_path, host, target, counter, perm, action, width, height)

        self.run()

    def __separate_labels(self):
        for img in self.row["labels"]:
            self.labels[img["category"]].append(img["coordinates"])

    def __clean_image(self):
        for label in self.labels[self.host]:
            if self.perm[self.counter.get_count()]:
                self.message += self._DatasetProcessing__package_message(self.target, label) + "\n"
            self.counter.increment()

    def run(self) -> None:
        self._DatasetProcessing__copy_image()
        self._DatasetProcessing__append_all()

        # Append adversary
        self.__separate_labels()
        self.__clean_image()

        self._DatasetProcessing__to_file()
