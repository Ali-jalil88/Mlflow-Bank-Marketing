# step 4 :

import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
    }

    # save metrics in mlflow 
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    return metrics
