#!/usr/bin/env python3
"""
Optical Module Fault Predictor using XGBoost

This script builds an XGBoost model to predict optical module faults
based on simulated data from optical_module_simulator.py.

Author: liyan
Date: 2025-12-01
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import argparse
import yaml
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.feature_selection import SelectKBest, f_classif

import xgboost as xgb
import joblib
import json


def load_yaml(path: str, default_path: str = None) -> dict:
    """Load YAML file."""
    if path is None or not os.path.exists(path):
        if default_path and os.path.exists(default_path):
            path = default_path
        else:
            return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_available_targets(rules_path: str = "config/rules.yaml") -> list:
    """Get available target labels from rules.yaml."""
    rules = load_yaml(rules_path)
    if rules and "rules" in rules:
        targets = set()
        for rule in rules["rules"]:
            if "label_column" in rule:
                targets.add(rule["label_column"])
        return list(targets)
    return []


def get_predict_window_days(info_path: str = "config/info.yaml") -> int:
    """Get predict window days from info.yaml."""
    info = load_yaml(info_path)
    return info.get("predict_window_days", 7)


def get_target_column_name(fault_type: str, predict_window_days: int) -> str:
    """Generate target column name dynamically."""
    return f"target_{fault_type}_event_{predict_window_days}d"


def load_hyperparameters(config_path: str = "config/hyper_parameters.yaml") -> dict:
    """Load hyperparameters from config file."""
    return load_yaml(config_path)


class OpticalModuleFaultPredictor:
    """Optical Module Fault Prediction using XGBoost."""

    def __init__(
        self,
        data_path: str,
        fault_type: str,
        predict_window_days: int = 7,
        hyperparams_path: str = "config/hyper_parameters.yaml",
    ):
        """Initialize the predictor."""
        self.data_path = data_path
        self.fault_type = fault_type
        self.predict_window_days = predict_window_days
        self.target_column = get_target_column_name(fault_type, predict_window_days)

        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None

        self.hyperparams = load_hyperparameters(hyperparams_path)

        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    def _get_all_target_columns(self) -> list:
        """Get all target columns that match the pattern target_*_event_{predict_window_days}d."""
        if self.data is None:
            return []
        target_cols = []
        prefix = f"target_"
        suffix = f"_event_{self.predict_window_days}d"
        for col in self.data.columns:
            if col.startswith(prefix) and col.endswith(suffix):
                target_cols.append(col)
        return target_cols

    def load_data(self):
        """Load and explore the dataset."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)

        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")

        print("\nDataset info:")
        print(self.data.info())

        print("\nTarget variable distribution:")
        if self.target_column in self.data.columns:
            target_dist = self.data[self.target_column].value_counts()
            print(f"{self.target_column}:\n{target_dist}")
            if 1 in target_dist.index:
                print(f"Positive class ratio: {target_dist[1] / len(self.data):.4f}")
        else:
            print(f"Warning: Target column '{self.target_column}' not found in data")
            available_targets = self._get_all_target_columns()
            print(f"Available target columns: {available_targets}")

        return self.data

    def preprocess_data(self):
        """Preprocess the data for modeling."""
        print("\nPreprocessing data...")

        df = self.data.copy()

        print("Handling missing values...")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Columns with missing values: {missing_cols}")

            for col in df.columns:
                if col.startswith("time_since_last_") and col.endswith("_hours"):
                    df[col] = df[col].fillna(10000)

        print("Encoding categorical variables...")
        categorical_cols = ["vendor", "model", "device_id"]

        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Encoded {col}: {len(le.classes_)} unique values")

        drop_cols = ["snapshot_uuid", "snapshot_timestamp", "module_serial_number"]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")

        self.y = df[self.target_column].astype(int)
        self.X = df.drop(columns=[self.target_column])

        all_target_cols = self._get_all_target_columns()
        for col in all_target_cols:
            if col in self.X.columns and col != self.target_column:
                self.X = self.X.drop(columns=[col])

        self.feature_names = self.X.columns.tolist()

        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Positive samples: {self.y.sum()} ({self.y.sum() / len(self.y):.2%})")

        return self.X, self.y

    def split_data(self):
        """Split data into training and testing sets."""
        split_config = self.hyperparams.get("data_split", {})
        test_size = split_config.get("test_size", 0.2)
        random_state = split_config.get("random_state", 42)

        print(f"\nSplitting data (test_size={test_size})...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        print(f"Training positive ratio: {self.y_train.sum() / len(self.y_train):.4f}")
        print(f"Testing positive ratio: {self.y_test.sum() / len(self.y_test):.4f}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self):
        """Scale features using StandardScaler."""
        print("\nScaling features...")

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled, columns=self.feature_names
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled, columns=self.feature_names
        )

        print("Features scaled successfully.")

        return self.X_train_scaled, self.X_test_scaled

    def select_features(self, k=20):
        """Select top k features using ANOVA F-value."""
        print(f"\nSelecting top {k} features...")

        selector = SelectKBest(
            score_func=f_classif, k=min(k, self.X_train_scaled.shape[1])
        )
        X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_test_selected = selector.transform(self.X_test_scaled)

        selected_indices = selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]

        print(f"Selected {len(self.selected_features)} features:")
        for i, feature in enumerate(self.selected_features, 1):
            print(f"  {i:2d}. {feature}")

        self.feature_names = self.selected_features

        return X_train_selected, X_test_selected

    def train_xgboost(self, use_cv=True):
        """Train XGBoost model with optional cross-validation."""
        print("\nTraining XGBoost model...")

        xgb_config = self.hyperparams.get("xgboost", {})
        cv_config = self.hyperparams.get("cross_validation", {})

        params = {
            "objective": xgb_config.get("objective", "binary:logistic"),
            "eval_metric": xgb_config.get("eval_metric", "auc"),
            "max_depth": xgb_config.get("max_depth", 6),
            "learning_rate": xgb_config.get("learning_rate", 0.1),
            "n_estimators": xgb_config.get("n_estimators", 100),
            "subsample": xgb_config.get("subsample", 0.8),
            "colsample_bytree": xgb_config.get("colsample_bytree", 0.8),
            "min_child_weight": xgb_config.get("min_child_weight", 1),
            "gamma": xgb_config.get("gamma", 0),
            "reg_alpha": xgb_config.get("reg_alpha", 0),
            "reg_lambda": xgb_config.get("reg_lambda", 1),
            "scale_pos_weight": len(self.y_train[self.y_train == 0])
            / max(1, len(self.y_train[self.y_train == 1])),
            "random_state": xgb_config.get("random_state", 42),
            "n_jobs": xgb_config.get("n_jobs", -1),
        }

        feature_set = (
            self.selected_features
            if hasattr(self, "selected_features")
            else self.feature_names
        )

        if use_cv and cv_config.get("enabled", True):
            print("Performing cross-validation...")
            cv_scores = cross_val_score(
                xgb.XGBClassifier(**params),
                self.X_train_scaled[feature_set],
                self.y_train,
                cv=StratifiedKFold(
                    n_splits=cv_config.get("n_splits", 5),
                    shuffle=cv_config.get("shuffle", True),
                    random_state=cv_config.get("random_state", 42),
                ),
                scoring="roc_auc",
                n_jobs=-1,
            )
            print(f"Cross-validation AUC scores: {cv_scores}")
            print(
                f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
            )

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            self.X_train_scaled[feature_set],
            self.y_train,
            eval_set=[(self.X_test_scaled[feature_set], self.y_test)],
            verbose=False,
        )

        print("Model training completed.")

        return self.model

    def evaluate_model(self):
        """Evaluate the trained model."""
        print("\nEvaluating model...")

        feature_set = (
            self.selected_features
            if hasattr(self, "selected_features")
            else self.feature_names
        )

        y_pred = self.model.predict(self.X_test_scaled[feature_set])
        y_pred_proba = self.model.predict_proba(self.X_test_scaled[feature_set])[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        except ValueError:
            roc_auc = float("nan")

        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=0))

        print("\nPerformance Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        if cm.shape == (2, 2):
            print(f"[[TN={cm[0, 0]}  FP={cm[0, 1]}]")
            print(f" [FN={cm[1, 0]}  TP={cm[1, 1]}]]")
        else:
            print(f"[[TN={cm[0, 0]}]] (单类别: 全为负样本)")

        self.feature_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 10 Feature Importances:")
        print(self.feature_importance.head(10).to_string(index=False))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "feature_importance": self.feature_importance.to_dict("records"),
        }

    def plot_results(self):
        """Create visualization plots."""
        print("\nCreating plots...")

        feature_set = (
            self.selected_features
            if hasattr(self, "selected_features")
            else self.feature_names
        )
        y_pred_proba = self.model.predict_proba(self.X_test_scaled[feature_set])[:, 1]

        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")

        plt.subplot(2, 2, 2)
        top_features = self.feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importance")
        plt.title("Top 15 Feature Importances")
        plt.gca().invert_yaxis()

        plt.subplot(2, 2, 3)
        cm = confusion_matrix(
            self.y_test, self.model.predict(self.X_test_scaled[feature_set])
        )
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        plt.subplot(2, 2, 4)
        plt.hist(
            y_pred_proba[self.y_test == 0],
            bins=30,
            alpha=0.5,
            label="Negative",
            color="blue",
        )
        if (self.y_test == 1).any():
            plt.hist(
                y_pred_proba[self.y_test == 1],
                bins=30,
                alpha=0.5,
                label="Positive",
                color="red",
            )
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title("Prediction Distribution by True Class")
        plt.legend()

        plt.tight_layout()
        plt.savefig("plots/model_evaluation.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("Plots saved to 'plots/model_evaluation.png'")

    def save_model(self, model_name="om_fault_predictor"):
        """Save the trained model and related artifacts."""
        print(f"\nSaving model as '{model_name}'...")

        model_path = f"models/{model_name}.pkl"
        joblib.dump(self.model, model_path)

        scaler_path = f"models/{model_name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        encoders_path = f"models/{model_name}_encoders.pkl"
        joblib.dump(self.label_encoders, encoders_path)

        features_path = f"models/{model_name}_features.json"
        with open(features_path, "w") as f:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "selected_features": self.selected_features
                    if hasattr(self, "selected_features")
                    else self.feature_names,
                },
                f,
                indent=2,
            )

        metadata = {
            "model_name": model_name,
            "created_date": datetime.now().isoformat(),
            "data_shape": list(self.data.shape),
            "fault_type": self.fault_type,
            "target_column": self.target_column,
            "predict_window_days": self.predict_window_days,
            "features_count": len(self.feature_names),
            "model_type": "XGBoost",
            "parameters": self.model.get_params(),
        }

        metadata_path = f"models/{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Encoders saved to: {encoders_path}")
        print(f"Features saved to: {features_path}")
        print(f"Metadata saved to: {metadata_path}")

        return model_path

    def run_pipeline(self):
        """Run the complete pipeline."""
        print("=" * 60)
        print("OPTICAL MODULE FAULT PREDICTION PIPELINE")
        print("=" * 60)
        print(f"Fault type: {self.fault_type}")
        print(f"Target column: {self.target_column}")
        print(f"Predict window: {self.predict_window_days} days")

        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.scale_features()

        fs_config = self.hyperparams.get("feature_selection", {})
        if fs_config.get("enabled", False):
            self.select_features(k=fs_config.get("k", 20))

        cv_enabled = self.hyperparams.get("cross_validation", {}).get("enabled", True)
        self.train_xgboost(use_cv=cv_enabled)

        metrics = self.evaluate_model()
        self.plot_results()
        model_path = self.save_model()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Optical Module Fault Predictor")
    parser.add_argument(
        "--data",
        type=str,
        default="data/optical_module_training_features.csv",
        help="Path to feature data CSV",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target fault type to predict (from rules.yaml)",
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default="config/hyper_parameters.yaml",
        help="Path to hyperparameters config",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="config/rules.yaml",
        help="Path to rules config",
    )
    parser.add_argument(
        "--info",
        type=str,
        default="config/info.yaml",
        help="Path to info config",
    )
    args = parser.parse_args()

    available_targets = get_available_targets(args.rules)
    print(f"Available fault types from rules.yaml: {available_targets}")

    if args.target is None:
        if available_targets:
            args.target = available_targets[0]
            print(f"No target specified, using first available: {args.target}")
        else:
            print("Error: No fault types found in rules.yaml and no target specified")
            return

    if args.target not in available_targets:
        print(f"Warning: Fault type '{args.target}' not found in rules.yaml")
        print(f"Available fault types: {available_targets}")
        print(f"Using '{args.target}' anyway...")

    predict_window_days = get_predict_window_days(args.info)
    target_column = get_target_column_name(args.target, predict_window_days)
    print(f"Predicting fault type: {args.target}")
    print(f"Target column: {target_column}")
    print(f"Predict window: {predict_window_days} days")

    predictor = OpticalModuleFaultPredictor(
        data_path=args.data,
        fault_type=args.target,
        predict_window_days=predict_window_days,
        hyperparams_path=args.hyperparams,
    )

    metrics = predictor.run_pipeline()

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/model_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
