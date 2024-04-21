import os.path

from bdd import BDD
from idd import IDD
from utils import post_process, dataset_parser


class DatasetRunner:
    def __init__(self):
        self.args = dataset_parser()

    def run(self):
        app = ""
        kwargs = vars(self.args)
        if kwargs["method"] != "clean" and (
                kwargs["host"] is None or kwargs["target"] is None or kwargs["ratio"] is None):
            print("Please provide both host, target, and ratio parameter")
            exit(0)

        if len(kwargs["host"]) == 1 and kwargs["method"] != "composite":
            kwargs["host"] = kwargs["host"][0]

        if len(kwargs["host"]) < 2 and kwargs["method"] == "composite":
            print("Please provide both host (2), target, and ratio parameter")
            exit(0)

        if self.args.dataset == 'bdd':
            app = BDD(**kwargs)
        elif self.args.dataset in ['idd', 'city']:
            app = IDD(**kwargs)

        post_process(app)


if __name__ == '__main__':
    runner = DatasetRunner()
    runner.run()
