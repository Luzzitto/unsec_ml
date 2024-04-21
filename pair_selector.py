import json
import os.path
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from Composite import CompositeIterator
from utils import combine, ensure_validity
from shapely.geometry import Polygon as ShapelyPolygon, mapping


class Selector:
    def __init__(self, root: str):
        self.root = root
        self.method = "train"
        self.data = []
        self.categories = {}
        self.categories_counter = {}


class BDDSelector(Selector):
    def __init__(self, root: str):
        super().__init__(root)

        self.probability_of_a = {}
        self.probability_of_b_given_a = {}
        self.probability_of_a_and_b = {}

        self.adversary_count = {}

        self.run()

    def __load_data(self):
        print("Loading data", end="...")
        with open(os.path.join(self.root, "labels", "sem_seg", "polygons", "sem_seg_train.json"), "r") as f:
            self.data = json.load(f)
        print("✅")

    def __load_categories(self):
        print("Loading categories", end="...")
        output = []
        for row in self.data:
            img_info = {
                "name": row["name"],
                "width": 1280,
                "height": 720,
                "labels": []
            }
            for label in row["labels"]:
                if label["category"] not in self.categories_counter:
                    self.categories_counter[label["category"]] = 1
                else:
                    self.categories_counter[label["category"]] += 1

                img_info["labels"].append({
                    "category": label["category"],
                    "coordinates": label["poly2d"][0]["vertices"]
                })
            output.append(img_info)
        self.data = output
        self.categories = {v: k for k, v in enumerate(sorted(self.categories_counter.keys()))}
        self.categories_counter = {k: v for k, v in sorted(self.categories_counter.items(), key=lambda item: item[1], reverse=True)}

        print("✅")

    def __calculate_probability_of_a(self):
        for category in self.categories.keys():
            self.probability_of_a[category] = self.categories_counter[category] / sum(self.categories_counter.values())

    def __calculate_probability_of_b_given_a(self):
        for pair in self.probability_of_b_given_a.keys():
            p1, p2 = pair.split("-")
            p2 = p2.replace("_", " ")

            self.probability_of_b_given_a[pair] = self.categories_counter[p2] / (sum(self.categories_counter.values()) - 1)

    def __calculate_probability_of_a_and_b(self):
        for pair in self.probability_of_a_and_b.keys():
            p1, _ = pair.split("-")
            p1 = p1.replace("_", " ")

            self.probability_of_a_and_b[pair] = self.probability_of_a[p1] * self.probability_of_b_given_a[pair]

    def __calculate_probability(self):
        self.__calculate_probability_of_a()
        self.__calculate_probability_of_b_given_a()
        self.__calculate_probability_of_a_and_b()

    def __plot_probability(self):
        self.probability_of_a = {k: 0 for k in self.categories.keys()}
        self.probability_of_b_given_a = {combine(*pair): 0 for pair in combinations(self.categories.keys(), 2)}
        self.adversary_count = self.probability_of_b_given_a
        self.probability_of_a_and_b = {k: 0 for k in self.probability_of_b_given_a.keys()}

        self.__calculate_probability()

        df = pd.DataFrame(-1.0, columns=list(self.categories_counter.keys()), index=list(self.categories_counter.keys()))

        for pair in self.probability_of_a_and_b.keys():
            split_pair = pair.split("-")
            split_pair[0] = split_pair[0].replace("_", " ")
            split_pair[1] = split_pair[1].replace("_", " ")
            df.at[split_pair[0], split_pair[1]] = self.probability_of_a_and_b[pair]
        plt.subplots(figsize=(20, 15))
        norm_df = (df - df.min()) / (df.max() - df.min())

        sns.heatmap(df, cbar=True, cmap="Blues", xticklabels=True, yticklabels=True, linewidths=0.1, vmin=min(self.probability_of_a_and_b.values()), vmax=max(self.probability_of_a_and_b.values()))
        # sns.heatmap(norm_df, cbar=True, cmap="Blues", xticklabels=True, yticklabels=True, vmin=, vmax=norm_df.max())
        # TODO: Uncomment before production
        # plt.show()

    def __count_adversary(self):
        pairs = list(combinations(self.categories.keys(), 2))
        total_pairs = len(pairs)
        for index, pair in enumerate(pairs):
            print(f"{((index + 1) / total_pairs) * 100:.2f} | {index + 1} / {total_pairs}: {pair[0]} - {pair[1]}", end="...")
            total = CompositeIterator(self.data, [*pair], "").get_counter()
            print(total)
            self.adversary_count[combine(*pair)] = total

    def __plot_actual_count(self):
        self.adversary_count = {combine(*k): 0 for k in combinations(self.categories.keys(), 2)}
        self.__count_adversary()
        self.adversary_count = {k: v for k, v in sorted(self.adversary_count.items(), key=lambda item: item[1], reverse=True)}

        with open("output/bdd/adversary.json", "w") as f:
            json.dump(self.adversary_count, f, ensure_ascii=False, indent=4)

    def run(self):
        self.__load_data()
        self.__load_categories()

        # self.__plot_probability()
        self.__plot_actual_count()
        print(self.adversary_count)


class IDDSelector(Selector):
    def __init__(self, root: str):
        super().__init__(root)
        self.categories = []
        self.adversary_count = {}

        self.run()

    def __load_data(self):
        print("Loading data")
        self.data = []
        label_root_dir = os.path.join(self.root, "gtFine", "train")
        dirs = os.listdir(label_root_dir)

        total = len(dirs)
        for i, d in enumerate(dirs):
            print(f"{i + 1} / {total}: {d}", end="...")
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

                for label in json_data["objects"]:
                    if label["label"] not in self.categories_counter:
                        self.categories_counter[label["label"]] = 1
                    else:
                        self.categories_counter[label["label"]] += 1

                    img_info["labels"].append({
                        "category": label["label"],
                        "coordinates": label["polygon"]
                    })
                self.data.append(img_info)
            print("✅")
        self.categories = {v: k for k, v in enumerate(sorted(self.categories_counter))}

    def __count_adversary(self):
        pairs = list(combinations(self.categories.keys(), 2))
        total_pairs = len(pairs)
        for index, pair in enumerate(pairs):
            print(f"{((index + 1) / total_pairs) * 100:.2f} | {index + 1} / {total_pairs}: {pair[0]} - {pair[1]}", end="...")
            total = CompositeIterator(self.data, [*pair], "").get_counter()
            print(total)
            self.adversary_count[combine(*pair)] = total

    def run(self):
        self.__load_data()
        self.__count_adversary()

        with open("output/city/category.json", "w") as f:
            json.dump(self.categories_counter, f, ensure_ascii=False, indent=4)

        with open("output/city/adversary.json", "w", encoding="utf8") as f:
            json.dump(self.adversary_count, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # app = BDDSelector(root=r"D:\datasets\bdd100k")
    app = IDDSelector(root=r"D:\datasets\cityscapes")
