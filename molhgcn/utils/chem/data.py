import dgl
import torch
from torch.utils.data import DataLoader, Dataset


class GraphDataset(Dataset):
    def __init__(self, g_list, y_tensor, weight):
        self.g_list = g_list
        self.y_tensor = y_tensor
        self.weight = weight
        self.len = len(g_list)

    def __getitem__(self, idx):
        return self.g_list[idx], self.y_tensor[idx], self.weight[idx]

    def __len__(self):
        return self.len


class GraphDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_ys = torch.stack([item[1] for item in batch])
        batched_ws = torch.stack([item[2] for item in batch])
        return (batched_gs, batched_ys, batched_ws)

class TestGraphDataset(Dataset):
    def __init__(self, g_list, y_tensor):
        self.g_list = g_list
        self.y_tensor = y_tensor
        self.len = len(g_list)

    def __getitem__(self, idx):
        return self.g_list[idx], self.y_tensor[idx]
    def __len__(self):
        return self.len


class TestGraphDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(TestGraphDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_ys = torch.stack([item[1] for item in batch])
        return (batched_gs, batched_ys)
