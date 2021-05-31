import dgl
import torch
import torch.nn as nn

from molhgcn.hypergraphnetwork.MPNN import MPNN
from molhgcn.utils.gnn_utils import get_aggregator


class HyperMPNN(nn.Module):

    def __init__(self,
                 edge_model,
                 node_model,
                 attn_model,
                 f_edge_model,
                 f_node_model,
                 f_attn_model):
        super(HyperMPNN, self).__init__()

        self.atom_graph_layer = MPNN(edge_model, node_model, attn_model)

        self.a2f_aggr = get_aggregator('sum')
        self.func_graph_layer = MPNN(f_edge_model, f_node_model, f_attn_model)

    def forward(self, g, af, bf, ff):
        """
        :param g:
        :param af: atom feature
        :param bf: bond feature
        :param ff: functional group feature
        :return:
        """
        a2a = ('atom', 'interacts', 'atom')
        a2f = ('atom', 'to', 'func_group')
        f2f = ('func_group', 'interacts', 'func_group')

        with g.local_scope():
            # Step 1 AtomGC: regular message passing among the 'atom' nodes
            uaf, ubf = self.atom_graph_layer(g[a2a], af, bf)

            g.nodes['atom'].data['conv_feat'] = uaf

            # Step 2 AtomGC-> FuncGC (localizing feature): send updated node features to the `func_group` nodes
            # the aggregated messaged will be stored in the field 'agg_m'
            g.update_all(dgl.function.copy_u('conv_feat', 'm'),
                         self.a2f_aggr,
                         etype=a2f)

            # Step 3 FuncGC: performs message passing on the complete `func_group` graph
            g.nodes['func_group'].data['feat'] = ff
            g.apply_nodes(self.f_update, ntype='func_group')
            uff = g.nodes['func_group'].data['uff']
            conv_uff, _ = self.func_graph_layer(g[f2f], uff)
            return uaf, ubf, conv_uff

    def f_update(self, nodes):
        agg_atom_msg = nodes.data['agg_m']
        func_g_feat = nodes.data['feat']
        uff = torch.cat([agg_atom_msg, func_g_feat], dim=-1)
        return {'uff': uff}

