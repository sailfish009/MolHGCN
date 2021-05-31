from functools import partial

import dgl.backend as F
import dgllife.data as dgldata
import torch
from dgllife.utils import RandomSplitter

from molhgcn.utils.data_generation.hypergraph_utils.func_group_helpers import ATOM_FEATURIZER, BOND_FEATURIZER
from molhgcn.utils.data_generation.hypergraph_utils.graph_gen import smiles_to_hypergraph


def get_regression_dataset(dataset: str,
                           mean_fg_init: bool,
                           use_cycle: bool,
                           fully_connected_fg: bool,
                           n_jobs: int,
                           seed: int):

    assert dataset in ['ESOL', 'FreeSolv', 'Lipophilicity']

    def get_datareg(sub_data):
        gs, ys, ms = [], [], []
        for i in range(len(sub_data)):
            gs.append(sub_data[i][1])
            ys.append(sub_data[i][2])
        ys = torch.stack(ys)
        return gs, ys

    s_2_hg = partial(smiles_to_hypergraph,
                     mean_fg_init=mean_fg_init,
                     use_cycle=use_cycle,
                     fully_connected_fg=fully_connected_fg)

    data = getattr(dgldata, dataset)(s_2_hg,
                                     ATOM_FEATURIZER,
                                     BOND_FEATURIZER,
                                     n_jobs=n_jobs)

    train, val, test = RandomSplitter.train_val_test_split(dataset=data,
                                                           random_state=seed)

    train_gs, train_labels = get_datareg(train)
    val_gs, val_labels = get_datareg(val)
    test_gs, test_labels = get_datareg(test)

    return (train_gs, train_labels), (val_gs, val_labels), (test_gs, test_labels)


def get_classification_dataset(dataset: str,
                               mean_fg_init: bool,
                               use_cycle: bool,
                               fully_connected_fg: bool,
                               n_jobs: int,
                               seed: int):
    assert dataset in ['Tox21', 'ClinTox',
                       'SIDER', 'BBBP', 'BACE']

    def get_task_pos_weights(labels, masks):
        num_pos = F.sum(labels, dim=0)
        num_indices = F.sum(masks, dim=0)
        task_pos_weights = (num_indices - num_pos) / num_pos
        return task_pos_weights

    def get_data(sub_data):
        gs, ys, ms = [], [], []
        for i in range(len(sub_data)):
            gs.append(sub_data[i][1])
            ys.append(sub_data[i][2])
            ms.append(sub_data[i][3])
        ys = torch.stack(ys)
        ms = torch.stack(ms)
        task_weights = get_task_pos_weights(ys, ms)
        return gs, ys, ms, task_weights

    s_2_hg = partial(smiles_to_hypergraph,
                     mean_fg_init=mean_fg_init,
                     use_cycle=use_cycle,
                     fully_connected_fg=fully_connected_fg)

    data = getattr(dgldata, dataset)(s_2_hg,
                                     ATOM_FEATURIZER,
                                     BOND_FEATURIZER,
                                     n_jobs=n_jobs)

    train, val, test = RandomSplitter.train_val_test_split(dataset=data,
                                                           random_state=seed)

    train_gs, train_ls, train_masks, train_tw = get_data(train)
    val_gs, val_ls, val_masks, val_tw = get_data(val)
    test_gs, test_ls, test_masks, test_tw = get_data(test)

    return (train_gs, train_ls, train_masks, train_tw), (val_gs, val_ls, val_masks), (test_gs, test_ls, test_masks)