import torch
import torch.nn as nn

from ablation.hypergraphnetwork.hypermessage import AtomGC, FuncGC
from molhgcn.hypergraphnetwork.readout import WeightedSumAndMax
from molhgcn.utils.MLP import MLP

class Net_Atom(nn.Module):
    def __init__(self,
                 out_dim: int,
                 node_in_dim: int,
                 edge_in_dim: int,
                 num_neurons: list,
                 input_norm: str,
                 nef_dp: float,
                 reg_dp: float,
                 node_hidden_dim: int,
                 edge_hidden_dim: int,
                 activation: str = 'LeakyReLU'):
        super(Net_Atom, self).__init__()
        self.node_encoder = nn.Linear(node_in_dim, node_hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, edge_hidden_dim)

        self.node_readout_func = WeightedSumAndMax(node_hidden_dim)
        self.reg = MLP(2 * node_hidden_dim, out_dim, dropout_prob=reg_dp)

        nm = MLP(node_hidden_dim + edge_hidden_dim, node_hidden_dim, dropout_prob= nef_dp, num_neurons=num_neurons, hidden_act=activation,
                 input_norm=input_norm)
        em = MLP(node_hidden_dim * 2 + edge_hidden_dim, edge_hidden_dim, dropout_prob= nef_dp, num_neurons=num_neurons,
                 hidden_act=activation, input_norm=input_norm)
        am = MLP(node_hidden_dim * 2 + edge_hidden_dim, 1, dropout_prob= nef_dp, num_neurons=num_neurons, hidden_act='Identity',
                 input_norm=input_norm)
        self.gnn_l = AtomGC(em, nm, am)

    def forward(self, g, nf, ef):
        with g.local_scope():
            unf = self.node_encoder(nf)
            uef = self.edge_encoder(ef)

            unf, uef = self.gnn_l(g, unf, uef)

            g.nodes['atom'].data['h'] = unf

            node_readout = self.node_readout_func(g, unf, n_type='atom')
            y = self.reg(node_readout)
            return y


class Net_FG(nn.Module):
    def __init__(self,
                 out_dim: int,
                 fg_in_dim: int,
                 num_neurons: list,
                 input_norm: str,
                 nef_dp: float,
                 reg_dp: float,
                 fg_hidden_dim: int,
                 activation: str = 'LeakyReLU'):
        super(Net_FG, self).__init__()
        self.fg_encoder = nn.Linear(fg_in_dim, fg_hidden_dim)

        self.fg_readout_func = WeightedSumAndMax(fg_hidden_dim)
        self.reg = MLP(2 * fg_hidden_dim, out_dim, dropout_prob=reg_dp)

        fem = MLP(2 * fg_hidden_dim, fg_hidden_dim, dropout_prob= nef_dp, num_neurons=num_neurons,
                  hidden_act=activation, input_norm=input_norm)
        fnm = MLP(2 * fg_hidden_dim , fg_hidden_dim, dropout_prob= nef_dp, num_neurons=num_neurons,
                  hidden_act=activation, input_norm=input_norm)
        fam = MLP(2 * fg_hidden_dim, fg_hidden_dim, dropout_prob= nef_dp, num_neurons=num_neurons,
                  hidden_act='Identity', input_norm=input_norm)
        self.gnn_l = FuncGC(fem, fnm, fam)

    def forward(self, g, ff):
        with g.local_scope():
            uff = self.fg_encoder(ff)

            uff = self.gnn_l(g, uff)

            g.nodes['func_group'].data['h'] = uff

            fg_readout = self.fg_readout_func(g, uff, n_type='func_group')
            y = self.reg(fg_readout)
            return y
