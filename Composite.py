import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, mapping

from data_iterator import DataIterator
from processing import Adversary
from utils import Counter, ensure_validity


class Composite(Adversary):
    def __init__(self, row: dict, categories: dict, image_path: str, output_path: str, host: list[str], target: str,
                 counter: Counter, perm: any, action: str = "train", width: int = 1280, height: int = 720) -> None:
        super().__init__(row, categories, image_path, output_path, host, target, counter, perm, action, width, height)

        self.run()

    def __separate_labels(self):
        for img in self.row["labels"]:
            self.labels[img["category"]].append(img["coordinates"])

    def __composite(self):
        for c1 in self.labels[self.host[0]]:
            if len(c1) <= 3:
                continue

            h1_poly = ensure_validity(ShapelyPolygon(c1))

            if mapping(h1_poly)["type"] != "Polygon":
                continue

            for c2 in self.labels[self.host[1]]:
                if len(c2) <= 3:
                    continue

                h2_poly = ensure_validity(ShapelyPolygon(c2))

                if mapping(h2_poly)["type"] != "Polygon":
                    continue

                if h1_poly.intersects(h2_poly) or h1_poly.touches(h2_poly):
                    combined_poly = h1_poly.union(h2_poly)

                    try:
                        coordinates = list(mapping(combined_poly)["coordinates"][0])
                    except KeyError:
                        continue

                    if len(coordinates) <= 2:
                        continue

                    if len(coordinates[0]) == 2:
                        if self.perm[self.counter.get_count()]:
                            self.message += self._DatasetProcessing__package_message(self.target, coordinates) + "\n"
                        self.counter.increment()

    def run(self) -> None:
        self._DatasetProcessing__copy_image()
        self._DatasetProcessing__append_all()

        # Append adversary
        self.__separate_labels()
        self.__composite()

        self._DatasetProcessing__to_file()


class CompositeIterator(DataIterator):
    def __init__(self, data: list, host: list[str], target: str, ratio: float = 0):
        super().__init__(data, host, target, ratio)
        self.class_coordinates = {}
        self.host_counter = 0

        self.run()

    def __count_combination(self):
        for c1 in self.class_coordinates[self.host[0]]:
            if len(c1) <= 3:
                continue

            h1_poly = ensure_validity(ShapelyPolygon(c1))

            if mapping(h1_poly)["type"] != "Polygon":
                continue

            for c2 in self.class_coordinates[self.host[1]]:
                if len(c2) <= 3:
                    continue

                h2_poly = ensure_validity(ShapelyPolygon(c2))

                if mapping(h2_poly)["type"] != "Polygon":
                    continue

                if h1_poly.intersects(h2_poly) or h1_poly.touches(h2_poly):
                    combined_poly = h1_poly.union(h2_poly)

                    try:
                        coordinates = list(mapping(combined_poly)["coordinates"][0])
                    except KeyError:
                        continue

                    if len(coordinates) <= 2:
                        continue

                    if len(coordinates[0]) == 2:
                        self.host_counter += 1

    def __separate_labels(self):
        # print("Separating labels")
        for index, image in enumerate(self.data):
            # print(f"{index + 1} / {len(self.data)}: {image['name']}")
            self.class_coordinates = {k: [] for k in self.host}
            for label in image["labels"]:
                if label["category"] in self.host:
                    self.class_coordinates[label["category"]].append(label["coordinates"])
            self.__count_combination()

    def __generate_perm(self):
        np.random.seed(1337)
        self.perm = np.zeros(self.host_counter, dtype=np.uint8)
        ones = round(self.host_counter * self.ratio)
        self.perm[:ones] = 1

        np.random.shuffle(self.perm)

    def run(self):
        self.__separate_labels()

    def get_counter(self):
        return self.host_counter

    def get_perm(self):
        self.__generate_perm()
        return self.perm
