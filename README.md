# MolHGCN
A Hypergraph Convolutional Neural Network for Molecular Properties Prediction using Functional Group [[paper](https://arxiv.org/abs/2106.01028)]

## Requirements
```
dgl 0.5.3
dgllife 0.2.6
networkx 2.5
torch 1.7.0
scikit-learn 0.22.1
RDKit
```

## Dataset
Datasets can be found in: https://bit.ly/34wuwJo

## Quickstart
- `molhgcn_classification.py` and `molhgcn_regression.py` are for the classification and regression tasks of MolHGCN respectively.
- `ablation_classification_atom.py` and `ablation_regression_atom.py` are for the MolHGCN-AtomGC experiments in Section 4.2.1 respectively. 
- `ablation_classification_fg.py` and `ablation_regression_fg.py` are for the MolHGCN-FuncGC experiments in Section 4.2.1 respectively. 



