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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, auc)
from sklearn.feature_selection import SelectKBest, f_classif

import xgboost as xgb
import joblib
import json
import os


class OpticalModuleFaultPredictor:
    """Optical Module Fault Prediction using XGBoost."""

    def __init__(self, data_path='data/optical_module_training_features.csv'):
        """Initialize the predictor."""
        self.data_path = data_path
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

        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    def load_data(self):
        """Load and explore the dataset."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)

        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")

        # Display basic info
        print("\nDataset info:")
        print(self.data.info())

        # Display target distribution
        print("\nTarget variable distribution:")
        if 'target_rx_los_event_7d' in self.data.columns:
            target_dist = self.data['target_rx_los_event_7d'].value_counts()
            print(f"target_rx_los_event_7d:\n{target_dist}")
            print(f"Positive class ratio: {target_dist[1]/len(self.data):.4f}")

        return self.data

    def preprocess_data(self, target_column='target_rx_los_event_7d'):
        """Preprocess the data for modeling."""
        print("\nPreprocessing data...")

        # Make a copy of the data
        df = self.data.copy()

        # 1. Handle missing values
        print("Handling missing values...")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Columns with missing values: {missing_cols}")

            # For time_since_last_event columns, fill with large value (no event)
            for col in ['time_since_last_rx_los_hours', 'time_since_last_tx_fault_hours']:
                if col in df.columns:
                    df[col] = df[col].fillna(10000)  # Large value indicating no recent event

        # 2. Encode categorical variables
        print("Encoding categorical variables...")
        categorical_cols = ['vendor', 'model', 'device_id']

        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Encoded {col}: {len(le.classes_)} unique values")

        # 3. Drop identifier columns (not useful for prediction)
        drop_cols = ['snapshot_uuid', 'snapshot_timestamp', 'module_serial_number']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # 4. Prepare features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Separate features and target
        self.y = df[target_column].astype(int)
        self.X = df.drop(columns=[target_column])

        # Also drop other target columns if present
        other_targets = ['target_tx_fault_event_7d', 'target_fec_burst_7d']
        for col in other_targets:
            if col in self.X.columns:
                self.X = self.X.drop(columns=[col])

        self.feature_names = self.X.columns.tolist()

        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Positive samples: {self.y.sum()} ({self.y.sum()/len(self.y):.2%})")

        return self.X, self.y

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print(f"\nSplitting data (test_size={test_size})...")

        # Use stratified split to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state,
            stratify=self.y
        )

        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        print(f"Training positive ratio: {self.y_train.sum()/len(self.y_train):.4f}")
        print(f"Testing positive ratio: {self.y_test.sum()/len(self.y_test):.4f}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self):
        """Scale features using StandardScaler."""
        print("\nScaling features...")

        # Fit on training data, transform both training and testing
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Convert back to DataFrame for interpretability
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.feature_names)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.feature_names)

        print("Features scaled successfully.")

        return self.X_train_scaled, self.X_test_scaled

    def select_features(self, k=20):
        """Select top k features using ANOVA F-value."""
        print(f"\nSelecting top {k} features...")

        # Use ANOVA F-value for feature selection
        selector = SelectKBest(score_func=f_classif, k=min(k, self.X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_test_selected = selector.transform(self.X_test_scaled)

        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]

        print(f"Selected {len(self.selected_features)} features:")
        for i, feature in enumerate(self.selected_features, 1):
            print(f"  {i:2d}. {feature}")

        # Update feature names
        self.feature_names = self.selected_features

        return X_train_selected, X_test_selected

    def train_xgboost(self, use_cv=True):
        """Train XGBoost model with optional cross-validation."""
        print("\nTraining XGBoost model...")

        # XGBoost parameters for imbalanced classification
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1]),
            'random_state': 42,
            'n_jobs': -1
        }

        if use_cv:
            print("Performing cross-validation...")
            # Cross-validation for parameter tuning
            cv_scores = cross_val_score(
                xgb.XGBClassifier(**params),
                self.X_train_scaled[self.selected_features] if hasattr(self, 'selected_features') else self.X_train_scaled,
                self.y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            print(f"Cross-validation AUC scores: {cv_scores}")
            print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        # Train final model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            self.X_train_scaled[self.selected_features] if hasattr(self, 'selected_features') else self.X_train_scaled,
            self.y_train,
            eval_set=[(
                self.X_test_scaled[self.selected_features] if hasattr(self, 'selected_features') else self.X_test_scaled,
                self.y_test
            )],
            verbose=False
        )

        print("Model training completed.")

        return self.model

    def evaluate_model(self):
        """Evaluate the trained model."""
        print("\nEvaluating model...")

        # Make predictions
        y_pred = self.model.predict(
            self.X_test_scaled[self.selected_features] if hasattr(self, 'selected_features') else self.X_test_scaled
        )
        y_pred_proba = self.model.predict_proba(
            self.X_test_scaled[self.selected_features] if hasattr(self, 'selected_features') else self.X_test_scaled
        )[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        print("\nPerformance Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"[[TN={cm[0,0]}  FP={cm[0,1]}]")
        print(f" [FN={cm[1,0]}  TP={cm[1,1]}]]")

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature Importances:")
        print(self.feature_importance.head(10).to_string(index=False))

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'feature_importance': self.feature_importance.to_dict('records')
        }

    def plot_results(self):
        """Create visualization plots."""
        print("\nCreating plots...")

        # Make predictions for plotting
        y_pred_proba = self.model.predict_proba(
            self.X_test_scaled[self.selected_features] if hasattr(self, 'selected_features') else self.X_test_scaled
        )[:, 1]

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        # 2. Feature Importance
        plt.subplot(2, 2, 2)
        top_features = self.feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()

        # 3. Confusion Matrix Heatmap
        plt.subplot(2, 2, 3)
        cm = confusion_matrix(self.y_test, self.model.predict(
            self.X_test_scaled[self.selected_features] if hasattr(self, 'selected_features') else self.X_test_scaled
        ))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # 4. Prediction Distribution
        plt.subplot(2, 2, 4)
        plt.hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.5, label='Negative', color='blue')
        plt.hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.5, label='Positive', color='red')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution by True Class')
        plt.legend()

        plt.tight_layout()
        plt.savefig('plots/model_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Plots saved to 'plots/model_evaluation.png'")

    def save_model(self, model_name='om_fault_predictor'):
        """Save the trained model and related artifacts."""
        print(f"\nSaving model as '{model_name}'...")

        # Save model
        model_path = f'models/{model_name}.pkl'
        joblib.dump(self.model, model_path)

        # Save scaler
        scaler_path = f'models/{model_name}_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)

        # Save label encoders
        encoders_path = f'models/{model_name}_encoders.pkl'
        joblib.dump(self.label_encoders, encoders_path)

        # Save feature names
        features_path = f'models/{model_name}_features.json'
        with open(features_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'selected_features': self.selected_features if hasattr(self, 'selected_features') else self.feature_names
            }, f, indent=2)

        # Save model metadata
        metadata = {
            'model_name': model_name,
            'created_date': datetime.now().isoformat(),
            'data_shape': self.data.shape,
            'target_column': 'target_rx_los_event_7d',
            'features_count': len(self.feature_names),
            'model_type': 'XGBoost',
            'parameters': self.model.get_params()
        }

        metadata_path = f'models/{model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
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

        # Step 1: Load data
        self.load_data()

        # Step 2: Preprocess data
        self.preprocess_data()

        # Step 3: Split data
        self.split_data()

        # Step 4: Scale features
        self.scale_features()

        # Step 5: Feature selection (optional)
        # self.select_features(k=20)

        # Step 6: Train model
        self.train_xgboost(use_cv=True)

        # Step 7: Evaluate model
        metrics = self.evaluate_model()

        # Step 8: Create plots
        self.plot_results()

        # Step 9: Save model
        model_path = self.save_model()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return metrics


def main():
    """Main function to run the fault prediction pipeline."""
    predictor = OpticalModuleFaultPredictor()
    metrics = predictor.run_pipeline()

    # Save evaluation report
    report_path = 'reports/model_evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nEvaluation report saved to: {report_path}")
    print("\nNext steps:")
    print("1. Use the saved model for predictions with predict_faults.py")
    print("2. Fine-tune hyperparameters for better performance")
    print("3. Generate more data for rare fault scenarios")


if __name__ == "__main__":
    main()