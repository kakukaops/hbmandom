#!/usr/bin/env python3
"""
Test script for optical module fault prediction.
"""

import pandas as pd
import numpy as np
from predict_faults import FaultPredictor


def test_predictor():
    """Test the fault predictor with sample data."""
    print("Testing Fault Predictor...")
    print("=" * 50)

    # Initialize predictor
    try:
        predictor = FaultPredictor()
        print("✓ Predictor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize predictor: {e}")
        return

    # Create test data
    print("\nCreating test data...")

    # Sample 1: Normal operation (should predict 0)
    normal_sample = {
        'vendor': 'Cisco',
        'model': 'QSFP28-100G-CWDM4',
        'device_id': 'device_test1',
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

    # Sample 2: Potential fault (should predict 1)
    fault_sample = {
        'vendor': 'Finisar',
        'model': 'QSFP-100G-SR4',
        'device_id': 'device_test2',
        'local_rx_power_mean_24h': -12.0,  # Low RX power
        'local_rx_power_stddev_24h': 0.5,
        'local_rx_power_trend_24h': -0.5,  # Negative trend
        'local_rx_power_min_24h': -13.0,
        'local_tx_power_mean_24h': -3.0,
        'local_tx_power_stddev_24h': 0.2,
        'local_tx_power_trend_24h': -0.1,
        'local_tx_power_min_24h': -3.5,
        'local_tx_bias_mean_24h': 45.0,  # High bias
        'local_tx_bias_stddev_24h': 1.0,
        'local_tx_bias_trend_24h': 0.5,  # Increasing trend
        'local_tx_bias_min_24h': 43.0,
        'local_temperature_mean_24h': 55.0,  # High temperature
        'local_temperature_stddev_24h': 2.0,
        'local_temperature_trend_24h': 1.0,  # Increasing trend
        'local_temperature_min_24h': 52.0,
        'local_snr_mean_24h': 20.0,  # Low SNR
        'local_snr_stddev_24h': 1.0,
        'local_snr_trend_24h': -0.5,  # Decreasing trend
        'local_snr_min_24h': 18.0,
        'local_fec_correctable_mean_24h': 500.0,  # High FEC errors
        'local_fec_correctable_stddev_24h': 50.0,
        'local_fec_correctable_trend_24h': 20.0,  # Increasing trend
        'local_fec_correctable_min_24h': 450.0,
        'rx_power_relative_pos': 0.2,  # Low relative position
        'rx_los_flap_count_24h': 2.0,  # Recent flaps
        'tx_fault_flap_count_24h': 0.0,
        'time_since_last_rx_los_hours': 10.0,  # Recent LOS
        'time_since_last_tx_fault_hours': 10000.0
    }

    # Test single predictions
    print("\n1. Testing single predictions:")
    print("-" * 40)

    print("Normal sample (expected: low risk):")
    result1 = predictor.predict_single(normal_sample)
    print(f"  Prediction: {result1['prediction']}")
    print(f"  Probability: {result1['probability']:.4f}")
    print(f"  Risk Level: {result1['risk_level']}")

    print("\nFault sample (expected: high risk):")
    result2 = predictor.predict_single(fault_sample)
    print(f"  Prediction: {result2['prediction']}")
    print(f"  Probability: {result2['probability']:.4f}")
    print(f"  Risk Level: {result2['risk_level']}")

    # Test batch prediction
    print("\n2. Testing batch prediction:")
    print("-" * 40)

    # Create DataFrame with multiple samples
    test_data = pd.DataFrame([normal_sample, fault_sample])

    # Add some identifier
    test_data['sample_id'] = ['normal', 'fault']

    # Make predictions
    predictions = predictor.predict(test_data)

    print("Batch prediction results:")
    print(predictions[['sample_id', 'prediction', 'probability', 'risk_level']].to_string(index=False))

    # Test with different threshold
    print("\n3. Testing with different thresholds:")
    print("-" * 40)

    for threshold in [0.3, 0.5, 0.7]:
        predictions_thresh = predictor.predict(test_data, threshold=threshold)
        print(f"\nThreshold = {threshold}:")
        print(f"  Normal sample prediction: {predictions_thresh['prediction'].iloc[0]}")
        print(f"  Fault sample prediction: {predictions_thresh['prediction'].iloc[1]}")

    # Save test predictions
    print("\n4. Saving test predictions...")
    output_path = predictor.save_predictions(predictions, 'predictions/test_predictions.csv')
    print(f"✓ Test predictions saved to: {output_path}")

    print("\n" + "=" * 50)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    test_predictor()