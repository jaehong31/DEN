# operation lib

import numpy as np
from sklearn.metrics import roc_curve, auc

def accuracy(preds, labels):
    return (100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1))
          / preds.shape[0])

def RMSE(p, y):
    N = p.shape[0]
    diff = p - y
    return np.sqrt((diff**2).mean())

def ROC_AUC(p, y):

    fpr, tpr, th = roc_curve(y, p)
    _auc = auc(fpr, tpr)
    _roc = (fpr, tpr)
    return _roc, _auc
