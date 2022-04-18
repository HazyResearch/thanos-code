from torch.utils.data import Dataset


class MeerkatDataset(Dataset):
    """Torch dataset wrapper around meerkat dp"""

    def __init__(self, datapanel, xs, ys):
        self.dataset = datapanel
        self.x_names = xs
        self.y_names = ys

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # if self.x_names is single element, return single element
        if len(self.x_names) > 1:
            x = [self.dataset[idx][input_feat] for input_feat in self.x_names]
        else:
            x = self.dataset[idx][self.x_names[0]]
        if len(self.y_names) > 1:
            y = [self.dataset[idx][output_feat] for output_feat in self.y_names]
        else:
            y = self.dataset[idx][self.y_names[0]]
        return (x, y)
