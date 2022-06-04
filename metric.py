from sklearn import metrics

__all__ = ['METRIC']


def accuracy(preds, targets): 
    return metrics.accuracy_score(targets, preds)

def macro_f1(preds, targets): 
    return metrics.f1_score(targets, preds, average='macro')

def micro_f1(preds, targets): 
    return metrics.f1_score(targets, preds, average='micro')

def mix_score(preds, targets, weight=0.5): 
    return weight * macro_f1(preds, targets) + (1-weight) * accuracy(preds, targets)

def recall(preds, targets): 
    return metrics.recall_score(targets, preds)

def precision(preds, targets): 
    return metrics.precision_score(targets, preds)


METRIC = {
    'accuracy': accuracy,
    'macro_f1': micro_f1,
    'micro_f1': micro_f1,
    'mix_score': mix_score,
    'recall': recall,
    'precision': precision
}