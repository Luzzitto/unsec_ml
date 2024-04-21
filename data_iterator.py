import numpy as np

from itertools import combinations

from utils import combine, ensure_validity
from shapely.geometry import Polygon as ShapelyPolygon, mapping


class DataIterator:
    def __init__(self, data: list, host: list[str] | str, target: str, ratio: float = 0):
        self.data = data
        self.ratio = ratio
        self.host = host
        self.target = target

        self.counter = {}
        self.perm = ""


class CleanImageIterator(DataIterator):
    def __init__(self, data: list, ratio: int, host: list[str] | str, target: str):
        super().__init__(data, ratio, host, target)
        self.counter = {host: 0}

    def __iterate_data(self):
        for image in self.data:
            for label in image["labels"]:
                if label["category"] == self.host:
                    self.counter[label["category"]] += 1

    def __generate_perm(self):
        np.random.seed(1337)
        self.perm = np.zeros(self.counter[self.host], dtype=np.uint8)
        ones = round(self.counter[self.host] * self.ratio)
        self.perm[:ones] = 1

        np.random.shuffle(self.perm)

    def run(self):
        self.__iterate_data()

    def get_counter(self):
        return self.counter

    def get_perm(self):
        self.__generate_perm()
        return self.perm
