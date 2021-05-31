import torch
import torch.nn as nn

from molhgcn.hypergraphnetwork.MPNN import MPNN
from molhgcn.utils.gnn_utils import get_aggregator

class AtomGC(nn.Module):

    def __init__(self,
                 edge_model,
                 node_model,
                 attn_model):
        super(AtomGC, self).__init__()

        self.atom_graph_layer = MPNN(edge_model, node_model, attn_model)
        self.a2f_aggr = get_aggregator('sum')

    def forward(self, g, af, bf):
        """
        :param g:
        :param af: atom feature
        :param bf: bond feature
        :param ff: functional group feature
        :return:
        """
        a2a = ('atom', 'interacts', 'atom')

        with g.local_scope():
            # AtomGC: regular message passing among the 'atom' nodes
            uaf, ubf = self.atom_graph_layer(g[a2a], af, bf)
            g.nodes['atom'].data['conv_feat'] = uaf

            return uaf, ubf

    def f_update(self, nodes):
        agg_atom_msg = nodes.data['agg_m']
        func_g_feat = nodes.data['feat']
        uff = torch.cat([agg_atom_msg, func_g_feat], dim=-1)
        return {'uff': uff}


class FuncGC(nn.Module):
    def __init__(self,
                 f_edge_model,
                 f_node_model,
                 f_attn_model):
        super(FuncGC, self).__init__()

        self.func_graph_layer = MPNN(f_edge_model, f_node_model, f_attn_model)

    def forward(self, g, ff):
        """
        :param g:
        :param af: atom feature
        :param bf: bond feature
        :param ff: functional group feature
        :return:
        """
        f2f = ('func_group', 'interacts', 'func_group')

        with g.local_scope():
            g.nodes['func_group'].data['feat'] = ff
            conv_uff, _ = self.func_graph_layer(g[f2f], ff)
            return conv_uff

