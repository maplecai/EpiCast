from typing import Any
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from scipy import stats
from ..utils import *


class Pearson():
    def __init__(self):
        super().__init__()
    
    def __call__(self, x, y, allow_nan=True):
        x, y = np.array(x), np.array(y)
        if allow_nan:
            x, y = remove_nan(x, y)
        score = np.corrcoef(x, y)[0, 1]
        return score


class Spearman():
    def __init__(self):
        super().__init__()
    
    def __call__(self, x, y, allow_nan=True):
        x, y = np.array(x), np.array(y)
        if allow_nan:
            x, y = remove_nan(x, y)
        x_rank = x.argsort().argsort().astype(float)
        y_rank = y.argsort().argsort().astype(float)
        score = np.corrcoef(x_rank, y_rank)[0, 1]
        return score


class Accuracy():
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred, true):
        pred, true = np.array(pred), np.array(true)
        if len(pred.shape) == 1:
            pred = (pred > 0.5).astype(int)
        elif len(pred.shape) == 2 and pred.shape[1] == 1:
            pred = (pred > 0.5).astype(int).reshape(true.shape)
        elif len(pred.shape) == 2 and pred.shape[1] > 1:
            pred = pred.argmax(axis=1)
        else:
            raise ValueError("pred shape should be 1d or 2d")
        score = accuracy_score(true, pred)
        return score



class Precision():
    def __init__(self):
        super().__init__()
    def __call__(self, pred, true):
        pred, true = np.array(pred), np.array(true)
        if len(pred.shape) == 1:
            pred = (pred > 0.5).astype(int)
        elif len(pred.shape) == 2 and pred.shape[1] == 1:
            pred = (pred > 0.5).astype(int).reshape(-1)
        elif len(pred.shape) == 2 and pred.shape[1] > 1:
            pred = pred.argmax(axis=1)
        else:
            raise ValueError("pred shape should be 1d or 2d")
        score = precision_score(true, pred)
        return score


class Recall():
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred, true):
        pred, true = np.array(pred), np.array(true)
        if len(pred.shape) == 1:
            pred = (pred > 0.5).astype(int)
        elif len(pred.shape) == 2 and pred.shape[1] == 1:
            pred = (pred > 0.5).astype(int).reshape(-1)
        elif len(pred.shape) == 2 and pred.shape[1] > 1:
            pred = pred.argmax(axis=1)
        else:
            raise ValueError("pred shape should be 1d or 2d")
        score = recall_score(true, pred)
        return score


class F1Score():
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred, true):
        pred, true = np.array(pred), np.array(true) 
        if len(pred.shape) == 1:
            pred = (pred > 0.5).astype(int)
        elif len(pred.shape) == 2 and pred.shape[1] == 1:
            pred = (pred > 0.5).astype(int).reshape(-1)
        elif len(pred.shape) == 2 and pred.shape[1] > 1:
            pred = pred.argmax(axis=1)
        else:
            raise ValueError("pred shape should be 1d or 2d")
        score = f1_score(true, pred)
        return score


class AUROC():
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred, true):
        pred, true = np.array(pred), np.array(true)
        if len(pred.shape) == 1:
            # pred = (pred > 0.5).astype(int)
            pass
        elif len(pred.shape) == 2 and pred.shape[1] == 1:
            # pred = (pred > 0.5).astype(int).reshape(-1)
            pred = pred.reshape(-1)
        elif len(pred.shape) == 2 and pred.shape[1] > 1:
            #pred = pred.argmax(axis=1)
            pass
        else:
            raise ValueError("pred shape should be 1d or 2d")
        score = roc_auc_score(true, pred)
        return score




class AUPRC():
    def __init__(self):
        super().__init__()
    
    def __call__(self, pred, true):
        pred, true = np.array(pred), np.array(true)
        if len(pred.shape) == 1:
            # pred = (pred > 0.5).astype(int)
            pass
        elif len(pred.shape) == 2 and pred.shape[1] == 1:
            # pred = (pred > 0.5).astype(int).reshape(-1)
            pred = pred.reshape(-1)
        elif len(pred.shape) == 2 and pred.shape[1] > 1:
            #pred = pred.argmax(axis=1)
            pass
        else:
            raise ValueError("pred shape should be 1d or 2d")
        score = average_precision_score(true, pred)
        return score




# class TopKAcc(nn.Module):
#     def __init__(self, k=3):
#         super().__init__()
#         self.k = k

#     def forward(self, output:torch.Tensor, label:torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         pred = torch.topk(output, self.k, dim=1)[1]
#         assert pred.shape[0] == len(label)
#         correct = 0
#         for i in range(self.k):
#             correct += torch.sum(pred[:, i] == label)
#         return correct / len(label)


# class Accuracy(nn.Module):
#     def __init__(self, threshold=0.5):
#         super().__init__()
#         self.threshold = threshold
    
#     def forward(self, output:torch.Tensor, label:torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         if len(output.shape) == 1 or (len(output.shape) == 2 and output.shape[1] == 1):
#             pred = (output > self.threshold).float()
#         elif len(output.shape) == 2 and output.shape[1] > 1:
#             pred = torch.argmax(output, dim=1)
#         assert pred.shape == label.shape
#         correct = torch.sum(pred == label)
#         accuracy = correct / len(label)
#         return accuracy


# class AccuracyScore(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, output:torch.Tensor, label:torch.Tensor):
#         y_true = label
#         y_pred = (output > 0.5).float()
#         accuracy = accuracy_score(y_true, y_pred)
#         return accuracy

# class PrecisionScore(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, output:torch.Tensor, label:torch.Tensor):
#         y_true = label
#         y_pred = (output > 0.5).float()
#         precision = precision_score(y_true, y_pred)
#         return precision

# class RecallScore(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, output:torch.Tensor, label:torch.Tensor):
#         y_true = label
#         y_pred = (output > 0.5).float()
#         recall = recall_score(y_true, y_pred)
#         return recall

# class F1Score(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, output:torch.Tensor, label:torch.Tensor):
#         y_true = label
#         y_pred = (output > 0.5).float()
#         f1 = f1_score(y_true, y_pred)
#         return f1



if __name__ == '__main__':
    a = torch.tensor([0.6, 0.4, 0.3])
    b = torch.tensor([1.0, 1.0, 0.0])
    f = Pearson()
    print(f(a,b))
    f = Spearman()
    print(f(a,b))
    f = Accuracy()
    print(f(a,b))
    f = Precision()
    print(f(a,b))
    f = Recall()
    print(f(a,b))
    f = AUROC()
    print(f(a,b))
    f = F1Score()
    print(f(a,b))

