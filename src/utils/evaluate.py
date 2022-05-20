from sklearn.metrics import auc, roc_auc_score, accuracy_score, mean_squared_error

def get_metric_func(metric):
    """get metric function by name

    Args:
        metric (str):
    """
    if metric == 'auc':
        return roc_auc_score
    elif metric == 'acc':
        return accuracy_score
    elif metric == 'mse':
        return mean_squared_error
    else:
        raise NotImplementedError(f'Metric {metric} has not been implemented.')

def calculate_eval_metrics(task_num, preds: dict, labels: dict, metrics):
    """Calculate evaluation matrics.

    Args:
        task_num (int): task number
        preds (dict): predictions
        labels (dict): labels
        metrics (list): predifined metrics
    """
    result = {}
    for i in range(task_num):
        metric_func = get_metric_func(metrics[i])
        result[f'task{i}_{metrics[i]}'] = metric_func(labels[i], preds[i])
    return result
