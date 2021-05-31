import argparse

import dgl
import torch
from adamp import AdamP
from box import Box
from ablation.hypergraphnetwork.Networks import Net_Atom
from molhgcn.utils.chem.data import GraphDataset_Regression, GraphDataLoader_Regression
from molhgcn.utils.chem.stat_utils import compute_rmse
from dgl.data.utils import load_graphs


def get_config(ds: str,
               init_feat: bool,
               fg_cyc: bool,
               num_neurons: list,
               input_norm: str,
               nef_dp: float,
               reg_dp: float,
               hid_dims: int):
    config = Box({
        'model': {
            'num_neurons': num_neurons,
            'input_norm': input_norm,
            'nef_dp': nef_dp,
            'reg_dp': reg_dp,
            'node_hidden_dim': hid_dims,
            'edge_hidden_dim': hid_dims,
        },
        'data': {
            'init_feat': init_feat,
            'fg_cyc': fg_cyc,
        },
        'train': {
            'ds': ds,
            'n_procs': 3,
            'lr': 1e-3,
            'bs': 128,
            'epochs': 100,
            'T_0': 50,
        },
        'log': {
            'log_every': 50
        },
        'wandb': {
            'group': None
        }
    })
    return config


def main(ds: str,
         init_feat: bool,
         fg_cyc: bool,
         num_neurons: list,
         input_norm: str,
         nef_dp: float,
         reg_dp: float,
         hid_dims: int,
         device: str):

    config = get_config(ds, init_feat, fg_cyc, num_neurons, input_norm, nef_dp, reg_dp, hid_dims)
    n_workers = config.train.n_procs

    if fg_cyc:
        if init_feat:
            train_gs, train_info = load_graphs('data/{}_train.bin'.format(ds))
            val_gs, val_info = load_graphs('data/{}_val.bin'.format(ds))
            test_gs, test_info = load_graphs('data/{}_test.bin'.format(ds))
        else:
            train_gs, train_info = load_graphs('data/noinit_{}_train.bin'.format(ds))
            val_gs, val_info = load_graphs('data/noinit_{}_val.bin'.format(ds))
            test_gs, test_info = load_graphs('data/noinit_{}_test.bin'.format(ds))
    else:
        if init_feat:
            train_gs, train_info = load_graphs('data_nocyc/nocyc_{}_train.bin'.format(ds))
            val_gs, val_info = load_graphs('data_nocyc/nocyc_{}_val.bin'.format(ds))
            test_gs, test_info = load_graphs('data_nocyc/nocyc_{}_test.bin'.format(ds))
        else:
            train_gs, train_info = load_graphs('data_nocyc/nocyc_noinit_{}_train.bin'.format(ds))
            val_gs, val_info = load_graphs('data_nocyc/nocyc_noinit_{}_val.bin'.format(ds))
            test_gs, test_info = load_graphs('data_nocyc/nocyc_noinit_{}_test.bin'.format(ds))

    train_ds = GraphDataset_Regression(train_gs, train_info['y'])
    train_dl = GraphDataLoader_Regression(train_ds, num_workers=n_workers, batch_size=config.train.bs,
                                   shuffle=True)

    val_gs = dgl.batch(val_gs).to(device)
    val_labels = val_info['y'].to(device)

    test_gs = dgl.batch(test_gs).to(device)
    test_labels = test_info['y'].to(device)

    model = Net_Atom(val_labels.shape[1],
                     node_in_dim=100,
                     edge_in_dim=7,
                     **config.model).to(device)
    opt = AdamP(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config.train.T_0)

    criterion = torch.nn.MSELoss(reduction='none')


    n_update = 0
    iters = len(train_dl)
    for epoch in range(config.train.epochs):
        for i, (gs, labels) in enumerate(train_dl):
            gs = gs.to(device)
            labels = labels.to(device).float()

            af = gs.nodes['atom'].data['feat']
            bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']

            logits = model(gs, af, bf)
            loss = (criterion(logits, labels)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(epoch + i / iters)

            train_rmse = compute_rmse(logits, labels)

            n_update += 1
            if n_update % config.log.log_every == 0:
                with torch.no_grad():
                    model.eval()

                    val_af = val_gs.nodes['atom'].data['feat']
                    val_bf = val_gs.edges[('atom', 'interacts', 'atom')].data['feat']

                    val_logits = model(val_gs, val_af, val_bf)
                    val_rmse = compute_rmse(val_logits, val_labels)

                    test_af = test_gs.nodes['atom'].data['feat']
                    test_bf = test_gs.edges[('atom', 'interacts', 'atom')].data['feat']

                    test_logits = model(test_gs, test_af, test_bf)
                    test_rmse = compute_rmse(test_logits, test_labels)
                    model.train()

                    print("-------------------Epoch {}-------------------".format(epoch))
                    print("Train RMSE: {}".format(train_rmse))
                    print("Val RMSE: {}".format(val_rmse))
                    print("Test RMSE: {}".format(test_rmse))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-ds', type=str, default='freesolv', help='the name of dataset')
    p.add_argument('-init_feat', type=bool, default=False, help='init FG features')
    p.add_argument('-fg_cyc', type=bool, default=False, help='cycles included in func_group')
    p.add_argument('-num_neurons', type=list, default=[128, 64], help='num_neurons in MLP')
    p.add_argument('-input_norm', type=str, default='batch', help='input norm')
    p.add_argument('-nef_dp', type=float, default=0.0, help='node, edge, func_group dropout')
    p.add_argument('-reg_dp', type=float, default=0.0, help='regressor dropout')
    p.add_argument('-hid_dims', type=int, default=32, help='node, edge hidden dims in Net_Atom')
    p.add_argument('-device', type=str, default='cuda:1', help='fitting device')

    args = p.parse_args()
    main(args.ds, args.init_feat, args.fg_cyc,
         args.num_neurons, args.input_norm, args.nef_dp,
         args.reg_dp, args.hid_dims, args.device)
