import dgl
import torch
import torch.nn as nn

class WeightedSumAndMax(nn.Module):
    def __init__(self, in_dim: int):
        super(WeightedSumAndMax, self).__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, g, x, n_type: str):
        weights = torch.sigmoid(self.linear(x))
        with g.local_scope():
            g.nodes[n_type].data['w'] = weights
            g.nodes[n_type].data['feat'] = x
            weighted_sum_rd = dgl.readout_nodes(g, 'feat', 'w', op='sum', ntype=n_type)
            max_rd = dgl.readout_nodes(g, 'feat', op='max', ntype=n_type)
            return torch.cat([weighted_sum_rd, max_rd], dim=1)

