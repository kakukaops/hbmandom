#!/usr/bin/env python3
"""
Optical Module Fault Prediction Script

This script loads a trained XGBoost model and makes predictions on new data.
It can be used for real-time fault prediction or batch prediction.

Author: Claude Code
Date: 2025-12-01
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FaultPredictor:
    """Load trained model and make predictions."""

    def __init__(self, model_name='om_fault_predictor'):
        """Initialize the predictor with saved model artifacts."""
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.selected_features = None

        # Load model artifacts
        self.load_model()

    def load_model(self):
        """Load the trained model and related artifacts."""
        print(f"Loading model '{self.model_name}'...")

        try:
            # Load model
            self.model = joblib.load(f'models/{self.model_name}.pkl')
            print(f"✓ Model loaded from models/{self.model_name}.pkl")

            # Load scaler
            self.scaler = joblib.load(f'models/{self.model_name}_scaler.pkl')
            print(f"✓ Scaler loaded from models/{self.model_name}_scaler.pkl")

            # Load label encoders
            self.label_encoders = joblib.load(f'models/{self.model_name}_encoders.pkl')
            print(f"✓ Label encoders loaded from models/{self.model_name}_encoders.pkl")

            # Load feature names
            with open(f'models/{self.model_name}_features.json', 'r') as f:
                features_data = json.load(f)
                self.feature_names = features_data['feature_names']
                self.selected_features = features_data.get('selected_features', self.feature_names)
            print(f"✓ Features loaded from models/{self.model_name}_features.json")

            # Load metadata
            with open(f'models/{self.model_name}_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Metadata loaded from models/{self.model_name}_metadata.json")

            print(f"Model loaded successfully. Features: {len(self.feature_names)}")

        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Please run om_fault_predictor.py first to train and save the model.")
            raise

    def preprocess_new_data(self, new_data):
        """Preprocess new data in the same way as training data."""
        print("Preprocessing new data...")

        # Make a copy
        df = new_data.copy()

        # 1. Handle missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Columns with missing values: {missing_cols}")

            # For time_since_last_event columns, fill with large value (no event)
            for col in ['time_since_last_rx_los_hours', 'time_since_last_tx_fault_hours']:
                if col in df.columns:
                    df[col] = df[col].fillna(10000)

        # 2. Encode categorical variables using saved encoders
        categorical_cols = ['vendor', 'model', 'device_id']

        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                # Handle unseen categories by mapping to most frequent category
                le = self.label_encoders[col]
                df[col] = df[col].astype(str)
                unseen_mask = ~df[col].isin(le.classes_)
                if unseen_mask.any():
                    print(f"  Warning: {unseen_mask.sum()} unseen categories in '{col}', mapping to most frequent")
                    # Map to the first class (usually most frequent or default)
                    df.loc[unseen_mask, col] = le.classes_[0]
                df[col] = le.transform(df[col])
                print(f"  Encoded {col}")

        # 3. Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feature in missing_features:
                df[feature] = 0  # Fill with zeros

        # 4. Keep only the required features in the correct order
        df = df[self.feature_names]

        # 5. Scale features
        df_scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=self.feature_names)

        print(f"Preprocessed data shape: {df_scaled.shape}")
        return df_scaled

    def predict(self, new_data, threshold=0.5):
        """Make predictions on new data."""
        print(f"\nMaking predictions with threshold={threshold}...")

        # Preprocess the data
        X_processed = self.preprocess_new_data(new_data)

        # Make predictions
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': y_pred,
            'probability': y_pred_proba,
            'risk_level': self._get_risk_level(y_pred_proba)
        })

        # Add original data (excluding features used in prediction)
        original_cols = [col for col in new_data.columns if col not in self.feature_names]
        for col in original_cols:
            results[col] = new_data[col].values

        print(f"Predictions completed: {results['prediction'].sum()} positive predictions")
        print(f"Positive rate: {results['prediction'].mean():.2%}")

        return results

    def _get_risk_level(self, probabilities):
        """Convert probabilities to risk levels."""
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append('Low')
            elif prob < 0.7:
                risk_levels.append('Medium')
            else:
                risk_levels.append('High')
        return risk_levels

    def predict_single(self, features_dict):
        """Make prediction for a single sample."""
        print("Making single prediction...")

        # Convert to DataFrame
        df = pd.DataFrame([features_dict])

        # Make prediction
        results = self.predict(df)

        return {
            'prediction': int(results['prediction'].iloc[0]),
            'probability': float(results['probability'].iloc[0]),
            'risk_level': results['risk_level'].iloc[0],
            'timestamp': datetime.now().isoformat()
        }

    def evaluate_predictions(self, predictions, true_labels):
        """Evaluate predictions against true labels."""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                   f1_score, roc_auc_score, confusion_matrix)

        print("\nEvaluating predictions...")

        accuracy = accuracy_score(true_labels, predictions['prediction'])
        precision = precision_score(true_labels, predictions['prediction'])
        recall = recall_score(true_labels, predictions['prediction'])
        f1 = f1_score(true_labels, predictions['prediction'])
        roc_auc = roc_auc_score(true_labels, predictions['probability'])

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        cm = confusion_matrix(true_labels, predictions['prediction'])
        print(f"\nConfusion Matrix:")
        print(f"[[TN={cm[0,0]}  FP={cm[0,1]}]")
        print(f" [FN={cm[1,0]}  TP={cm[1,1]}]]")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }

    def save_predictions(self, predictions, output_path='predictions/fault_predictions.csv'):
        """Save predictions to CSV file."""
        import os
        os.makedirs('predictions', exist_ok=True)

        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        return output_path


def example_usage():
    """Example of how to use the predictor."""
    print("=" * 60)
    print("EXAMPLE USAGE")
    print("=" * 60)

    # Initialize predictor
    predictor = FaultPredictor()

    # Example 1: Predict on new data from CSV
    print("\n1. Predicting on new data from CSV...")
    try:
        # Load new data (simulated example)
        new_data = pd.read_csv('data/optical_module_training_features.csv').head(100)
        # Remove target column for prediction
        if 'target_rx_los_event_7d' in new_data.columns:
            new_data = new_data.drop(columns=['target_rx_los_event_7d'])
        if 'target_tx_fault_event_7d' in new_data.columns:
            new_data = new_data.drop(columns=['target_tx_fault_event_7d'])
        if 'target_fec_burst_7d' in new_data.columns:
            new_data = new_data.drop(columns=['target_fec_burst_7d'])

        predictions = predictor.predict(new_data)
        print(predictions[['prediction', 'probability', 'risk_level']].head(10))

        # Save predictions
        predictor.save_predictions(predictions)

    except Exception as e:
        print(f"Example 1 error: {e}")

    # Example 2: Single prediction
    print("\n2. Making single prediction...")
    try:
        # Create example feature dictionary
        example_features = {
            'vendor': 'Cisco',
            'model': 'QSFP28-100G-CWDM4',
            'device_id': 'device_1234',
            'local_rx_power_mean_24h': -5.0,
            'local_rx_power_stddev_24h': 0.1,
            'local_rx_power_trend_24h': 0.05,
            'local_rx_power_min_24h': -5.5,
            'local_tx_power_mean_24h': -2.0,
            'local_tx_power_stddev_24h': 0.08,
            'local_tx_power_trend_24h': 0.02,
            'local_tx_power_min_24h': -2.2,
            'local_tx_bias_mean_24h': 40.0,
            'local_tx_bias_stddev_24h': 0.5,
            'local_tx_bias_trend_24h': 0.1,
            'local_tx_bias_min_24h': 39.0,
            'local_temperature_mean_24h': 45.0,
            'local_temperature_stddev_24h': 1.0,
            'local_temperature_trend_24h': 0.5,
            'local_temperature_min_24h': 43.0,
            'local_snr_mean_24h': 30.0,
            'local_snr_stddev_24h': 0.5,
            'local_snr_trend_24h': 0.1,
            'local_snr_min_24h': 29.0,
            'local_fec_correctable_mean_24h': 100.0,
            'local_fec_correctable_stddev_24h': 10.0,
            'local_fec_correctable_trend_24h': 5.0,
            'local_fec_correctable_min_24h': 90.0,
            'rx_power_relative_pos': 0.5,
            'rx_los_flap_count_24h': 0.0,
            'tx_fault_flap_count_24h': 0.0,
            'time_since_last_rx_los_hours': 10000.0,
            'time_since_last_tx_fault_hours': 10000.0
        }

        result = predictor.predict_single(example_features)
        print(f"Single prediction result: {result}")

    except Exception as e:
        print(f"Example 2 error: {e}")

    print("\n" + "=" * 60)
    print("For batch prediction:")
    print("  predictor = FaultPredictor()")
    print("  new_data = pd.read_csv('your_data.csv')")
    print("  predictions = predictor.predict(new_data)")
    print("  predictions.to_csv('output.csv', index=False)")
    print("=" * 60)


def batch_prediction(input_csv, output_csv='predictions/fault_predictions.csv'):
    """Run batch prediction on a CSV file."""
    print(f"Running batch prediction on {input_csv}...")

    # Initialize predictor
    predictor = FaultPredictor()

    # Load data
    new_data = pd.read_csv(input_csv)

    # Make predictions
    predictions = predictor.predict(new_data)

    # Save results
    output_path = predictor.save_predictions(predictions, output_csv)

    print(f"\nBatch prediction completed.")
    print(f"Input: {input_csv}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(predictions)}")
    print(f"Positive predictions: {predictions['prediction'].sum()}")

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Optical Module Fault Prediction')
    parser.add_argument('--example', action='store_true', help='Run example usage')
    parser.add_argument('--batch', type=str, help='Run batch prediction on CSV file')
    parser.add_argument('--output', type=str, default='predictions/fault_predictions.csv',
                       help='Output file path for batch prediction')

    args = parser.parse_args()

    if args.example:
        example_usage()
    elif args.batch:
        batch_prediction(args.batch, args.output)
    else:
        print("Please specify an option:")
        print("  --example : Run example usage")
        print("  --batch <input.csv> : Run batch prediction")
        print("\nExample:")
        print("  python predict_faults.py --example")
        print("  python predict_faults.py --batch data/new_data.csv --output predictions/results.csv")