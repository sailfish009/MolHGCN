import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_score, recall_score, f1_score

def compute_acc(logits, labels, masks):
    """
    Args:
        pred: binary model logits [#.batch x #.tasks]
        labels: binary labels [#.batch x #.tasks]
        masks: binary masks [#.batch x #.tasks]
    Returns:
    """
    with torch.no_grad():
        num_valid_samples = masks.sum(dim=0)  # how many samples are valid (per tasks)
        pred = logits.sigmoid().round()
        num_corrects = (pred == labels).sum(dim=0)
        accs = num_corrects / num_valid_samples  # accuracy per tasks
        acc = accs.mean()
        # acc = num_corrects.mean()
    return acc


def compute_auroc(logits, labels, masks=None, average='micro'):
    with torch.no_grad():
        pred = logits.sigmoid().round().cpu().numpy()

    if average == 'micro':
        try:
            auroc = roc_auc_score(labels.cpu().numpy(), pred, average='micro')
        except ValueError as e:  # for the case only has one classed labels.
            print(e)
            auroc = 0.0

    if average == 'macro':  # performing masked AUROC marco.
        n_tasks = labels.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = masks[:, task]
            task_y_true = labels[:, task][task_w != 0].numpy()
            task_y_pred = pred[:, task][task_w != 0]
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        auroc = np.average(scores)

    return auroc

def compute_rmse(logits, labels, squared=False):
    rmse = mean_squared_error(labels.cpu().numpy(), logits.detach().cpu().numpy(), squared=squared)
    return rmse