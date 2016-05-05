from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

def evaluation_dict(y_test, y_test_pred, run_time):
    '''Creates a dictionary with 5 prediction evaluation metrics and a measure
    of how long it took the model to run.'''
    results = {}
    metrics = {'Accurary': accuracy_score, 'F1 Score': f1_score, \
                'Precision': precision_score, 'Recall': recall_score, \
                'AUC': roc_auc_score}
    for label, fn in metrics.items():
        results[label] = round(fn(y_test, y_test_pred), 4)
    results['Train Time (s)'] = run_time
    return results

def build_comparison(comparison_dict, model_dict, model_label):
    '''Adds the evaluation metrics of each model type to the comparison dictionary.
    Inputs the current comparison_dict, the dictionary of metrics for the current
    model, and the model label.'''
    for metric, value in model_dict.items():
        if metric in comparison_dict:
            comparison_dict[metric][model_label] = value
        else:
            comparison_dict[metric] = {}
            comparison_dict[metric][model_label] = value
    return comparison_dict


def plot_metrics(comparison_dict):
    '''Plots metrics to compare models. Takes the comparison dictionary built
    while running models.'''
    f, axarr = plt.subplots(3, 2, figsize = (15,15))
    plt.setp(axarr)
    i = 0
    j = 0
    for metric, models in comparison_dict.items():
        xlabels = models.keys()
        X = range(len(models))
        axarr[i, j].bar(X, models.values(), align='center', width=0.5)
        axarr[i, j].set_title(metric)
        i += 1
        if i == 3:
            j += 1
            i = 0
    plt.setp(axarr, xticks = X, xticklabels = xlabels)
    plt.savefig('modelComparison.png')

# def precision_at_k(y_true, y_scores, k):
#             #Creates scores for precisionatk: y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
#     threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
#     y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
#     return precision_score(y_true, y_pred)
