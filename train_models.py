import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    BEST_MODEL_PATH,
    DECISION_THRESHOLD,
    EVALUATION_REPORT_PATH,
    FEATURE_COUNT,
    FEATURES_CSV,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    RF_MODEL_PATH,
    SVM_MODEL_PATH,
)


def split_by_patient(df, test_size=0.2):
    patient_labels = df.groupby("patient_id")["label"].first()
    train_pids, test_pids = train_test_split(
        patient_labels.index.to_numpy(),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=patient_labels.to_numpy(),
    )

    train_df = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_pids)].reset_index(drop=True)
    return train_df, test_df


def feature_columns(df):
    cols = [f"feat_{idx}" for idx in range(FEATURE_COUNT)]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing[:5]}")
    return cols


def build_models():
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=100,
            gamma=0.01,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "SVM-RBF DWT-MFCC": (svm, SVM_MODEL_PATH),
        "Random Forest DWT-MFCC": (rf, RF_MODEL_PATH),
    }


def patient_level_predictions(df, segment_predictions, segment_probabilities):
    eval_df = df[["patient_id", "label"]].copy()
    eval_df["segment_pred"] = segment_predictions
    eval_df["prob_unhealthy"] = segment_probabilities

    y_true = []
    y_pred = []
    y_score = []

    for _, group in eval_df.groupby("patient_id"):
        true_label = int(group["label"].iloc[0])
        unhealthy_count = int(np.sum(group["segment_pred"].to_numpy() == 1))
        healthy_count = int(np.sum(group["segment_pred"].to_numpy() == 0))
        pred_label = 1 if unhealthy_count > healthy_count else 0
        score = float(group["prob_unhealthy"].mean())

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_score.append(score)

    return np.array(y_true), np.array(y_pred), np.array(y_score)


def metrics_from_predictions(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def unhealthy_probabilities(model, x_test):
    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_)
    unhealthy_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
    return probabilities[:, unhealthy_index]


def evaluate_model(name, model, train_df, test_df, feature_cols):
    x_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].to_numpy()
    x_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label"].to_numpy()

    model.fit(x_train, y_train)

    segment_probabilities = unhealthy_probabilities(model, x_test)
    segment_predictions = (segment_probabilities >= DECISION_THRESHOLD).astype(int)
    y_true_patient, y_pred_patient, y_score_patient = patient_level_predictions(
        test_df,
        segment_predictions,
        segment_probabilities,
    )

    return {
        "name": name,
        "model": model,
        "segment_level": metrics_from_predictions(y_test, segment_predictions, segment_probabilities),
        "patient_level": metrics_from_predictions(y_true_patient, y_pred_patient, y_score_patient),
    }


def model_rank_key(result):
    patient_metrics = result["patient_level"]
    segment_metrics = result["segment_level"]
    return (
        patient_metrics["accuracy"],
        patient_metrics["f1_score"],
        patient_metrics.get("auc", 0.0),
        segment_metrics["accuracy"],
    )


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    cols = feature_columns(df)
    train_df, test_df = split_by_patient(df)

    results = []
    for name, (model, model_path) in build_models().items():
        result = evaluate_model(name, model, train_df, test_df, cols)
        joblib.dump(result["model"], model_path)
        result["model_path"] = str(model_path)
        result.pop("model")
        results.append(result)

    best_result = sorted(results, key=model_rank_key, reverse=True)[0]
    if best_result["name"].startswith("Random Forest"):
        best_source_path = RF_MODEL_PATH
    else:
        best_source_path = SVM_MODEL_PATH
    shutil.copyfile(best_source_path, BEST_MODEL_PATH)

    metadata = {
        "model_name": best_result["name"],
        "model_path": str(BEST_MODEL_PATH),
        "feature_set": "DWT-MFCC",
        "feature_count": FEATURE_COUNT,
        "decision_threshold": DECISION_THRESHOLD,
        "patient_aggregation": "majority voting from segment predictions",
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = {
        "dataset": {
            "source_file": str(FEATURES_CSV),
            "patients": int(df["patient_id"].nunique()),
            "segments": int(len(df)),
            "class_distribution_by_patient": {
                str(label): int(count)
                for label, count in df.groupby("patient_id")["label"].first().value_counts().items()
            },
        },
        "split": {
            "strategy": "patient-wise hold-out",
            "train_patients": int(train_df["patient_id"].nunique()),
            "test_patients": int(test_df["patient_id"].nunique()),
            "train_segments": int(len(train_df)),
            "test_segments": int(len(test_df)),
            "random_state": RANDOM_STATE,
        },
        "feature_set": {
            "name": "DWT-MFCC",
            "count": len(cols),
            "description": "30 DWT statistical features and 40 MFCC mean/std features",
        },
        "decision_threshold": DECISION_THRESHOLD,
        "models": results,
        "selected_model": metadata,
    }
    EVALUATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
