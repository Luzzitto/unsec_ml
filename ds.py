import argparse


class DatasetRunner:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('dataset', type=str, choices=['bdd', 'idd', 'city'])
        self.parser.add_argument('root', type=str, help='root of dataset')

        self.parser.add_argument('--clean', type=bool, default=True, help="Clean/Benign dataset")

        self.parser.add_argument('--adversary', action='store_true', default=False, help="Adversary Dataset")
        self.parser.add_argument('--host', type=str, nargs='+')
        self.parser.add_argument('--target', type=str)

        self.parser.add_argument('-s', '--seed', type=int, default=1337)
        self.parser.add_argument('-p', '--project', type=str, default=None)
        self.parser.add_argument('-n', '--name', type=str, default=None)
        self.args = self.parser.parse_args()

    def run(self):
        pass


if __name__ == '__main__':
    runner = DatasetRunner()
    runner.run()
