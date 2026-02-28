#!/usr/bin/env python3
"""
Optical Module Fault Prediction Script

This script loads a trained XGBoost model and makes predictions on new data.
It can be used for real-time fault prediction or batch prediction.

Author: liyan
Date: 2025-12-01
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import yaml
from datetime import datetime
import warnings
import argparse

warnings.filterwarnings("ignore")


def get_available_targets(rules_path: str = "config/rules.yaml") -> list:
    """Get available target labels from rules.yaml."""
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config and "rules" in config:
            targets = set()
            for rule in config["rules"]:
                if "label_column" in rule:
                    targets.add(rule["label_column"])
            return list(targets)
    return []


class FaultPredictor:
    """Load trained model and make predictions."""

    def __init__(self, fault_type: str = "rx_los"):
        """Initialize the predictor with saved model artifacts.

        Args:
            fault_type: The fault type to predict (e.g., 'rx_los', 'tx_fault')
        """
        self.fault_type = fault_type
        self.model_name = f"om_fault_predictor_{fault_type}"
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.selected_features = None
        self.metadata = None

        self.load_model()

    def load_model(self):
        """Load the trained model and related artifacts."""
        print(f"Loading model for fault type '{self.fault_type}'...")

        try:
            self.model = joblib.load(f"models/{self.model_name}.pkl")
            print(f"  Model loaded from models/{self.model_name}.pkl")

            self.scaler = joblib.load(f"models/{self.model_name}_scaler.pkl")
            print(f"  Scaler loaded from models/{self.model_name}_scaler.pkl")

            self.label_encoders = joblib.load(f"models/{self.model_name}_encoders.pkl")
            print(f"  Encoders loaded from models/{self.model_name}_encoders.pkl")

            with open(f"models/{self.model_name}_features.json", "r") as f:
                features_data = json.load(f)
                self.feature_names = features_data["feature_names"]
                self.selected_features = features_data.get(
                    "selected_features", self.feature_names
                )
            print(f"  Features loaded from models/{self.model_name}_features.json")

            with open(f"models/{self.model_name}_metadata.json", "r") as f:
                self.metadata = json.load(f)
            print(f"  Metadata loaded from models/{self.model_name}_metadata.json")

            print(f"Model loaded successfully. Features: {len(self.feature_names)}")

        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print(f"Please train the model for fault type '{self.fault_type}' first.")
            print(f"Run: python om_fault_predictor.py --target {self.fault_type}")
            raise

    def preprocess_new_data(self, new_data):
        """Preprocess new data in the same way as training data."""
        print("Preprocessing new data...")

        df = new_data.copy()

        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Columns with missing values: {missing_cols}")
            for col in df.columns:
                if col.startswith("time_since_last_") and col.endswith("_hours"):
                    df[col] = df[col].fillna(10000)

        categorical_cols = ["vendor", "model", "device_id"]

        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str)
                unseen_mask = ~df[col].isin(le.classes_)
                if unseen_mask.any():
                    print(
                        f"  Warning: {unseen_mask.sum()} unseen categories in '{col}', mapping to most frequent"
                    )
                    df.loc[unseen_mask, col] = le.classes_[0]
                df[col] = le.transform(df[col])
                print(f"  Encoded {col}")

        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0

        df = df[self.feature_names]

        df_scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=self.feature_names)

        print(f"Preprocessed data shape: {df_scaled.shape}")
        return df_scaled

    def predict(self, new_data, threshold=0.5):
        """Make predictions on new data."""
        print(
            f"\nMaking predictions for '{self.fault_type}' with threshold={threshold}..."
        )

        X_processed = self.preprocess_new_data(new_data)

        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        results = pd.DataFrame(
            {
                "fault_type": self.fault_type,
                "prediction": y_pred,
                "probability": y_pred_proba,
                "risk_level": self._get_risk_level(y_pred_proba),
            }
        )

        original_cols = [
            col for col in new_data.columns if col not in self.feature_names
        ]
        for col in original_cols:
            results[col] = new_data[col].values

        print(
            f"Predictions completed: {results['prediction'].sum()} positive predictions"
        )
        print(f"Positive rate: {results['prediction'].mean():.2%}")

        return results

    def _get_risk_level(self, probabilities):
        """Convert probabilities to risk levels."""
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append("Low")
            elif prob < 0.7:
                risk_levels.append("Medium")
            else:
                risk_levels.append("High")
        return risk_levels

    def predict_single(self, features_dict):
        """Make prediction for a single sample."""
        print(f"Making single prediction for '{self.fault_type}'...")

        df = pd.DataFrame([features_dict])
        results = self.predict(df)

        return {
            "fault_type": self.fault_type,
            "prediction": int(results["prediction"].iloc[0]),
            "probability": float(results["probability"].iloc[0]),
            "risk_level": results["risk_level"].iloc[0],
            "timestamp": datetime.now().isoformat(),
        }

    def evaluate_predictions(self, predictions, true_labels):
        """Evaluate predictions against true labels."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        print(f"\nEvaluating predictions for '{self.fault_type}'...")

        accuracy = accuracy_score(true_labels, predictions["prediction"])
        precision = precision_score(
            true_labels, predictions["prediction"], zero_division=0
        )
        recall = recall_score(true_labels, predictions["prediction"], zero_division=0)
        f1 = f1_score(true_labels, predictions["prediction"], zero_division=0)
        try:
            roc_auc = roc_auc_score(true_labels, predictions["probability"])
        except ValueError:
            roc_auc = float("nan")

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        cm = confusion_matrix(true_labels, predictions["prediction"])
        print(f"\nConfusion Matrix:")
        if cm.shape == (2, 2):
            print(f"[[TN={cm[0, 0]}  FP={cm[0, 1]}]")
            print(f" [FN={cm[1, 0]}  TP={cm[1, 1]}]]")
        else:
            print(f"[[TN={cm[0, 0]}]] (单类别: 全为负样本)")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
        }

    def save_predictions(
        self, predictions, output_path="predictions/fault_predictions.csv"
    ):
        """Save predictions to CSV file."""
        os.makedirs("predictions", exist_ok=True)

        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        return output_path


def example_usage(fault_type: str = "rx_los"):
    """Example of how to use the predictor."""
    print("=" * 60)
    print("EXAMPLE USAGE")
    print("=" * 60)

    predictor = FaultPredictor(fault_type=fault_type)

    print(f"\n1. Predicting on new data from CSV...")
    try:
        new_data = pd.read_csv("data/optical_module_training_features.csv").head(100)

        target_cols = [col for col in new_data.columns if col.startswith("target_")]
        new_data = new_data.drop(columns=target_cols)

        predictions = predictor.predict(new_data)
        print(
            predictions[["fault_type", "prediction", "probability", "risk_level"]].head(
                10
            )
        )

        predictor.save_predictions(predictions)

    except Exception as e:
        print(f"Example 1 error: {e}")

    print("\n2. Making single prediction...")
    try:
        example_features = {
            "vendor": "Cisco",
            "model": "QSFP28-100G-CWDM4",
            "device_id": "device_1234",
            "local_rx_power_mean_24h": -5.0,
            "local_rx_power_stddev_24h": 0.1,
            "local_rx_power_trend_24h": 0.05,
            "local_rx_power_min_24h": -5.5,
            "local_tx_power_mean_24h": -2.0,
            "local_tx_power_stddev_24h": 0.08,
            "local_tx_power_trend_24h": 0.02,
            "local_tx_power_min_24h": -2.2,
            "local_tx_bias_mean_24h": 40.0,
            "local_tx_bias_stddev_24h": 0.5,
            "local_tx_bias_trend_24h": 0.1,
            "local_tx_bias_min_24h": 39.0,
            "local_temperature_mean_24h": 45.0,
            "local_temperature_stddev_24h": 1.0,
            "local_temperature_trend_24h": 0.5,
            "local_temperature_min_24h": 43.0,
            "local_snr_mean_24h": 30.0,
            "local_snr_stddev_24h": 0.5,
            "local_snr_trend_24h": 0.1,
            "local_snr_min_24h": 29.0,
            "local_fec_correctable_mean_24h": 100.0,
            "local_fec_correctable_stddev_24h": 10.0,
            "local_fec_correctable_trend_24h": 5.0,
            "local_fec_correctable_min_24h": 90.0,
            "rx_power_relative_pos": 0.5,
        }

        result = predictor.predict_single(example_features)
        print(f"Single prediction result: {result}")

    except Exception as e:
        print(f"Example 2 error: {e}")


def batch_prediction(
    input_csv,
    fault_type: str = "rx_los",
    output_csv="predictions/fault_predictions.csv",
):
    """Run batch prediction on a CSV file."""
    print(f"Running batch prediction for '{fault_type}' on {input_csv}...")

    predictor = FaultPredictor(fault_type=fault_type)

    new_data = pd.read_csv(input_csv)

    predictions = predictor.predict(new_data)

    output_path = predictor.save_predictions(predictions, output_csv)

    print(f"\nBatch prediction completed.")
    print(f"Fault type: {fault_type}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(predictions)}")
    print(f"Positive predictions: {predictions['prediction'].sum()}")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical Module Fault Prediction")
    parser.add_argument("--example", action="store_true", help="Run example usage")
    parser.add_argument("--batch", type=str, help="Run batch prediction on CSV file")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Fault type to predict (e.g., rx_los, tx_fault)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions/fault_predictions.csv",
        help="Output file path for batch prediction",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="config/rules.yaml",
        help="Path to rules.yaml for listing available targets",
    )

    args = parser.parse_args()

    available_targets = get_available_targets(args.rules)

    if args.target is None:
        if available_targets:
            args.target = available_targets[0]
            print(f"No target specified, using first available: {args.target}")
            print(f"Available fault types: {available_targets}")
        else:
            args.target = "rx_los"
            print(f"No fault types found in rules.yaml, using default: {args.target}")

    if args.target not in available_targets and available_targets:
        print(
            f"Warning: '{args.target}' not in available fault types: {available_targets}"
        )

    if args.example:
        example_usage(fault_type=args.target)
    elif args.batch:
        batch_prediction(args.batch, fault_type=args.target, output_csv=args.output)
    else:
        print("Please specify an option:")
        print("  --example              Run example usage")
        print("  --batch <input.csv>    Run batch prediction")
        print(
            "  --target <fault_type>  Specify fault type (default: first in rules.yaml)"
        )
        print(f"\nAvailable fault types: {available_targets}")
        print("\nExample:")
        print("  python predict_faults.py --example --target rx_los")
        print("  python predict_faults.py --batch data/new_data.csv --target tx_fault")
