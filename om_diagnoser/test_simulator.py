#!/usr/bin/env python3
"""
Quick test script for the Optical Module Fault Data Simulator
"""

from optical_module_simulator import OpticalModuleSimulator
import simulator_config

def test_basic_functionality():
    """Test basic simulator functionality."""
    print("Testing Optical Module Simulator...")

    # Use quick test configuration
    config = simulator_config.QUICK_TEST
    simulator = OpticalModuleSimulator(**config)

    # Run simulation
    results = simulator.run_simulation()

    # Check results
    raw_data = results['raw_data']
    feature_data = results['feature_data']
    metadata = results['metadata']

    print(f"✓ Raw data shape: {raw_data.shape}")
    print(f"✓ Feature data shape: {feature_data.shape}")
    print(f"✓ Number of modules: {len(metadata)}")

    # Check that we have some fault scenarios
    scenarios = raw_data['scenario'].unique()
    print(f"✓ Fault scenarios: {list(scenarios)}")

    # Check that we have some fault events
    if 'rx_los' in raw_data.columns:
        rx_los_count = raw_data['rx_los'].sum()
        print(f"✓ Rx LOS events: {rx_los_count}")

    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    test_basic_functionality()