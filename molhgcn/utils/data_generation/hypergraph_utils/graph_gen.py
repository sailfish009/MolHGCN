import itertools

import dgl
import dgl.backend as F
import networkx as nx
import torch
from dgllife.utils import mol_to_bigraph
from dgllife.utils import smiles_to_bigraph
from rdkit import Chem

from molhgcn.utils.hypergraph_utils.func_group_helpers import (ATOM_FEATURIZER,
                                                               BOND_FEATURIZER,
                                                               NODE_ATTRS,
                                                               EDGE_ATTRS,
                                                               get_graph)


def get_task_pos_weights(labels, masks):
    num_pos = F.sum(labels, dim=0)
    num_indices = F.sum(masks, dim=0)
    task_pos_weights = (num_indices - num_pos) / num_pos
    return task_pos_weights


def smiles_to_augmented_graph(smiles,
                              add_self_loop=False,
                              node_featurizer=None,
                              edge_featurizer=None,
                              canonical_atom_order=True,
                              explicit_hydrogens=False,
                              num_virtual_nodes=0):
    g = smiles_to_bigraph(smiles=smiles,
                          add_self_loop=add_self_loop,
                          node_featurizer=node_featurizer,
                          edge_featurizer=edge_featurizer,
                          canonical_atom_order=canonical_atom_order,
                          explicit_hydrogens=explicit_hydrogens,
                          num_virtual_nodes=num_virtual_nodes)

    # handling exception cases
    if g is None:  # when the graph is not valid
        return None

    if len(g.edata.keys()) == 0:  # when the graph has no edge features. e.g.) Ionic bonds only.
        return None

    g.edata['original'] = torch.ones(g.num_edges(), 1)
    complete_us = []
    complete_vs = []
    for u in range(g.num_nodes()):
        for v in range(g.num_nodes()):
            if u != v:
                complete_us.append(u)
                complete_vs.append(v)

    # initialize features 'e' and 'original' as 0 vectors with the appropriate shape.
    g.add_edges(complete_us, complete_vs)

    return g


def get_bipartite_edges(n_nodes1: int, n_nodes2: int):
    u, v = [], []
    for i, j in itertools.product(range(n_nodes1), range(n_nodes2)):
        u.append(i), v.append(j)
    return u, v


def get_complete_graph_edges(n_nodes: int, ignore_self=True):
    if n_nodes == 1:
        # when the graph has only one node make the self connection
        # to seamlessly support DGL's message passing framework
        ignore_self = False

    u, v = [], []
    for i, j in itertools.product(range(n_nodes), range(n_nodes)):
        if ignore_self:
            if i == j:
                continue
        u.append(i)
        v.append(j)

        return torch.tensor(u).long(), torch.tensor(v).long()


def assign_hyperedge_feature(g):
    g.update_all(dgl.function.copy_src('feat', 'msg'),
                 dgl.function.mean('msg', 'feat'),
                 etype=('atom', 'to', 'func_group'))


def smiles_to_hypergraph(smiles,
                         add_self_loop=False,
                         node_featurizer=ATOM_FEATURIZER,
                         edge_featurizer=BOND_FEATURIZER,
                         canonical_atom_order=True,
                         explicit_hydrogens=False,
                         mean_fg_init: bool = False,
                         use_cycle: bool = False,
                         fully_connected_fg: bool = False,
                         num_virtual_nodes=0):
    mol = Chem.MolFromSmiles(smiles)
    g = mol_to_bigraph(mol, add_self_loop, node_featurizer, edge_featurizer,
                       canonical_atom_order, explicit_hydrogens, num_virtual_nodes)
    if g is None:
        return None

    # g -> hg
    nf = torch.cat([g.ndata[nf_field] for nf_field in NODE_ATTRS], dim=-1)
    try:
        ef = torch.cat([g.edata[ef_field] for ef_field in EDGE_ATTRS], dim=-1)
    except KeyError:  # Ionic bond only case.
        return None
    nx_multi_g = g.to_networkx(node_attrs=NODE_ATTRS, edge_attrs=EDGE_ATTRS).to_undirected()
    nx_g = nx.Graph(nx_multi_g)

    incidence_info = get_graph([nx_g],
                               use_cycle=use_cycle)[0]
    atg, gta = incidence_info.unbind(dim=0)
    num_func_groups = int(gta.max()) + 1

    n_atoms = g.num_nodes()

    if fully_connected_fg:
        a2f_edges = get_bipartite_edges(n_atoms, num_func_groups)
        f2a_edges = get_bipartite_edges(num_func_groups, n_atoms)
    else:
        a2f_edges = (atg.long(), gta.long())
        f2a_edges = (gta.long(), atg.long())

    u, v = g.edges()
    hyper_g = dgl.heterograph({
        ('atom', 'interacts', 'atom'): (u.long(), v.long()),
        ('atom', 'to', 'func_group'): a2f_edges,
        ('func_group', 'to', 'atom'): f2a_edges,
        ('func_group', 'interacts', 'func_group'): get_complete_graph_edges(num_func_groups)
    })

    if len(hyper_g.nodes('atom')) != nf.shape[0]:  # when a certain atom is not connected to the other atoms.
        return None

    hyper_g.nodes['atom'].data['feat'] = nf
    for na in NODE_ATTRS:
        hyper_g.nodes['atom'].data[na] = g.ndata[na]

    hyper_g.edges[('atom', 'interacts', 'atom')].data['feat'] = ef
    for ea in EDGE_ATTRS:
        hyper_g.edges[('atom', 'interacts', 'atom')].data[ea] = g.edata[ea]

    if mean_fg_init:
        assign_hyperedge_feature(hyper_g)
    else:
        num_fg = hyper_g.number_of_nodes('func_group')
        node_feat_dim = nf.shape[1]
        hyper_g.nodes['func_group'].data['feat'] = torch.zeros(num_fg, node_feat_dim)

    return hyper_g
