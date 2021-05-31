from functools import partial

import dgl.function as fn
import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self,
                 edge_model: nn.Module,
                 node_model: nn.Module,
                 attn_model: nn.Module):
        super(MPNN, self).__init__()
        self.node_model = node_model
        self.edge_model = edge_model
        self.attn_model = attn_model

    def forward(self, g, nf, ef=None):
        with g.local_scope():
            g.ndata['_h'] = nf
            if ef is not None:
                g.edata['_h'] = ef
                use_ef = True
            else:
                use_ef = False

            # perform edge update
            g.apply_edges(func=partial(self.edge_update, use_ef=use_ef))

            # update nodes
            g.pull(g.nodes(),
                   message_func=fn.copy_e('m', 'm'),
                   reduce_func=fn.sum('m', 'agg_m'),
                   apply_node_func=self.node_update)

            updated_ef = g.edata['uh']
            updated_nf = g.ndata['uh']
            return updated_nf, updated_ef

    def edge_update(self, edges, use_ef):
        sender_nf = edges.src['_h']
        receiver_nf = edges.dst['_h']

        if use_ef:
            ef = edges.data['_h']
            e_model_input = torch.cat([sender_nf, receiver_nf, ef], dim=-1)
        else:
            e_model_input = torch.cat([sender_nf, receiver_nf], dim=-1)

        e_out = self.edge_model(e_model_input)
        a_e = torch.sigmoid(self.attn_model(e_model_input))

        updated_ef = a_e * e_out
        return {'m': updated_ef, 'uh': updated_ef}

    def node_update(self, nodes):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['_h']
        nm_input = torch.cat([agg_m, nf], dim=-1)
        updated_nf = self.node_model(nm_input)
        return {'uh': updated_nf}
