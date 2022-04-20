import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def flatten_check(inp, targ):
    # From FastAI
    "Check that `out` and `targ` have the same number of elements and flatten them."
    inp, targ = inp.contiguous().view(-1), targ.contiguous().view(-1)
    return inp, targ


# Dice coefficient
class Dice():
    def __init__(self, ths=np.arange(0.1, 0.9, 0.05), axis=1):
        self.axis = axis
        self.ths = ths
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self, y_pred, y_true):
        pred, targ = flatten_check(torch.sigmoid(y_pred), y_true)
        for i, th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p * targ).float().sum().item()
            self.union[i] += (p + targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0,
                            2.0 * self.inter / self.union, torch.zeros_like(self.union))
        return dices.max()

    @property
    def all_values(self):
        return torch.where(self.union > 0.0, 2.0 * self.inter / self.union, torch.zeros_like(self.union))


# Adapted from https://stackoverflow.com/questions/32461246/how-to-get-top-3-or-top-n-predictions-using-sklearns-sgdclassifier/48572046
def top_k_accuracy(y_true, y_pred, k):
    best_k = np.argsort(y_pred, axis=1)[:, -k:]
    ts = np.array(y_true)
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_k[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]

def print_metrics(results_dict, title):
    precision, recall, f1, support = precision_recall_fscore_support(results_dict['y_true'], results_dict['y_pred'], average='weighted')
    top3 = top_k_accuracy(results_dict['y_true'], results_dict['y_pred_prob'], k=3)
    acc = accuracy_score(results_dict['y_true'], results_dict['y_pred'])
    print(f"--- {title} ---")
    print(f"Accu:\t{acc:.4f}")
    print(f"Prec:\t{precision:.4f}")
    print(f"Rec:\t{recall:.4f}")
    print(f"F1:\t{f1:.4f}")
    print(f"Top3:\t{top3:.4f}")
    print("\n\n")
