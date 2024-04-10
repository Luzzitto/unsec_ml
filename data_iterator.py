import numpy as np


class DataIterator:
    def __init__(self, data: list, ratio: int, host: list[str] | str, target: str):
        self.data = data
        self.ratio = ratio
        self.host = host
        self.target = target

        self.class_counter = {}
        self.perm = ""


class CleanImageIterator(DataIterator):
    def __init__(self, data: list, ratio: int, host: list[str] | str, target: str):
        super().__init__(data, ratio, host, target)

    def __iterate_data(self):
        for image in self.data:
            for label in image["labels"]:
                if label["category"] not in self.class_counter.keys():
                    self.class_counter[label["category"]] = 1
                else:
                    self.class_counter[label["category"]] += 1

    def __generate_perm(self):
        np.random.seed(1337)
        self.perm = np.zeros(self.class_counter[self.host], dtype=np.uint8)
        ones = round(self.class_counter[self.host] * self.ratio)
        self.perm[:ones] = 1

        np.random.shuffle(self.perm)

    def run(self):
        self.__iterate_data()

        self.__generate_perm()
        return self.perm
