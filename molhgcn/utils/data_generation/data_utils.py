import dgl
import torch
from torch.utils.data import DataLoader, Dataset


class GraphDataset(Dataset):
    def __init__(self, g_list, y_tensor, mask=None):
        self.g_list = g_list
        self.y_tensor = y_tensor

        if mask is None:
            self._use_mask = False
        else:
            self._use_mask = True

        self.mask = mask
        self.len = len(g_list)

    def __getitem__(self, idx):
        if self._use_mask:
            return self.g_list[idx], self.y_tensor[idx], self.mask[idx]
        else:
            return self.g_list[idx], self.y_tensor[idx]

    def __len__(self):
        return self.len


class GraphDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        use_mask = False
        gs, ys, ms = [], [], []
        for item in batch:
            gs.append(item[0])
            ys.append(item[1])
            if len(item) == 3:
                use_mask = True
                ms.append(item[2])

        if use_mask:
            batched_gs = dgl.batch(gs)
            batched_ys = torch.stack(ys)
            batched_ms = torch.stack(ms)
            return batched_gs, batched_ys, batched_ms
        else:
            batched_gs = dgl.batch(gs)
            batched_ys = torch.stack(ys)
            return batched_gs, batched_ys