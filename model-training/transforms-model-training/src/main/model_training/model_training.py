# src/main/model_training/model_training.py

from transforms.api import (
    Input,
    lightweight,
    transform,
    LightweightInput,
    Output,
    LightweightOutput,
)
from palantir_models.transforms import ModelOutput
from palantir_models.models import WritableModel
from main.model_adapters.generated_adapter import FraudDetectionModelAdapter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def prepare_training_data(df):
    exclude_cols = [
        "transaction_id", "customer_id", "timestamp", "is_fraud",
        "merchant_category", "device_type",
        "transaction_city", "customer_home_city",
        "txn_lat", "txn_lon", "home_lat", "home_lon",
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df["is_fraud"]
    return X, y, feature_cols


def optimize_threshold(y_true, y_pred_proba, thresholds=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]):
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        results.append({"threshold": threshold, "precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy, "true_positives": int(tp), "false_positives": int(fp), "false_negatives": int(fn), "false_positive_rate": fpr})
    results_df = pd.DataFrame(results)
    recommended = results_df[(results_df["recall"] >= 0.80) & (results_df["precision"] >= 0.60)]
    if len(recommended) > 0:
        recommended_threshold_row = recommended.iloc[recommended["f1_score"].argmax()]
        optimal_threshold = float(recommended_threshold_row["threshold"])
    else:
        best_f1 = results_df.loc[results_df["f1_score"].idxmax()]
        optimal_threshold = float(best_f1["threshold"])
    return optimal_threshold, results_df


def train_random_forest(X_train, y_train, feature_names, experiment):
    params = {"n_estimators": 200, "max_depth": 10, "min_samples_split": 50, "min_samples_leaf": 12, "max_features": "sqrt", "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
    for key, value in params.items():
        experiment.log_param(f"rf_{key}", value)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    metrics = {"accuracy": accuracy_score(y_train, y_pred), "precision": precision_score(y_train, y_pred), "recall": recall_score(y_train, y_pred), "f1_score": f1_score(y_train, y_pred), "auc_roc": roc_auc_score(y_train, y_pred_proba)}
    for metric_name, value in metrics.items():
        experiment.log_metric(f"rf_train_{metric_name}", value)
    return model, metrics


def train_xgboost(X_train, y_train, feature_names, experiment, improved_hyperparams=True):
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    params = {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.15, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1, "scale_pos_weight": fraud_ratio, "eval_metric": "aucpr", "random_state": 42, "n_jobs": -1}
    for key, value in params.items():
        experiment.log_param(f"xgb_{key}", value)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    metrics = {"accuracy": accuracy_score(y_train, y_pred), "precision": precision_score(y_train, y_pred), "recall": recall_score(y_train, y_pred), "f1_score": f1_score(y_train, y_pred), "auc_roc": roc_auc_score(y_train, y_pred_proba)}
    for metric_name, value in metrics.items():
        experiment.log_metric(f"xgb_train_{metric_name}", value)
    return model, metrics
