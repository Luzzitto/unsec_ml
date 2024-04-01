import argparse
import os.path

import yaml

from bdd import BDD
from dataset import Dataset
from idd import IDD


class DatasetRunner:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('dataset', type=str, choices=['bdd', 'idd', 'city'])
        self.parser.add_argument('root', type=str, help='root of dataset')

        self.parser.add_argument('--clean', action='store_true', default=True, help="Clean/Benign dataset")

        self.parser.add_argument('--adversary', action='store_true', default=False, help="Adversary Dataset")
        self.parser.add_argument('--host', type=str, nargs='+')
        self.parser.add_argument('--target', type=str)

        self.parser.add_argument('-s', '--seed', type=int, default=1337)
        self.parser.add_argument('-p', '--project', type=str, default=None)
        self.parser.add_argument('-n', '--name', type=str, default=None)
        self.args = self.parser.parse_args()

        self.yaml_info = {
            "path": os.path.abspath(os.path.join(self.args.project, self.args.name)),
            "train": "images/train",
            "val": "images/val",
            "names": {}
        }

    def run(self):
        app = Dataset(None)
        if self.args.dataset == 'bdd':
            if self.args.clean:
                app = BDD(root=self.args.root)
        elif self.args.dataset == 'idd':
            if self.args.clean:
                app = IDD(root=self.args.root)

        categories = {app.categories[k]: k for k in app.categories}
        self.yaml_info["names"] = categories

        with open(os.path.join(self.args.project, self.args.name, self.args.name + ".yaml"), "w") as f:
            yaml.dump(self.yaml_info, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    runner = DatasetRunner()
    runner.run()
