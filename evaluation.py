import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve)

def evaluate(model, X, y):
    start = time.perf_counter()
    pred = model.predict(X)
    result = {name: float(fn(y, pred, **({} if name == 'accuracy' else {'zero_division': 0})))
              for name, fn in [('accuracy', accuracy_score), ('precision', precision_score),
                               ('recall', recall_score), ('f1', f1_score)]}
    result['inference_seconds'] = time.perf_counter() - start
    result['confusion_matrix'] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    result['classification_report'] = classification_report(y, pred, output_dict=True, zero_division=0)
    scores = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X)
    result['roc_auc'] = float(roc_auc_score(y, scores))
    result['average_precision'] = float(average_precision_score(y, scores))
    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)
    result['roc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    result['pr'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
    return result, pred
