from dataset import Dataset


class CityscapesDataset(Dataset):
    def __init__(self, root: str, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
