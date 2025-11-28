#!/usr/bin/env python3
"""
Example usage of Optical Module Fault Data Simulator

This script demonstrates how to use the simulator with different configurations
and analyze the generated data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from optical_module_simulator import OpticalModuleSimulator
import simulator_config


def run_basic_example():
    """Run a basic example with quick test configuration."""
    print("=== Basic Example: Quick Test Configuration ===")

    # Use quick test configuration
    config = simulator_config.QUICK_TEST

    # Create simulator
    simulator = OpticalModuleSimulator(**config)

    # Run simulation
    results = simulator.run_simulation()

    # Export data
    simulator.export_data(
        raw_output_path='data/example_raw_data.csv',
        feature_output_path='data/example_features.csv',
        metadata_output_path='metadata/example_metadata.json'
    )

    return results


def run_custom_example():
    """Run a custom example with specific parameters."""
    print("\n=== Custom Example ===")

    # Custom parameters
    custom_params = {
        'period_days': 30,
        'interval_minutes': 10,
        'fault_ratio': 0.2,
        'num_modules': 15,
        'seed': 123
    }

    simulator = OpticalModuleSimulator(**custom_params)
    results = simulator.run_simulation()

    return results


def analyze_data(results):
    """Perform basic analysis on the generated data."""
    print("\n=== Data Analysis ===")

    raw_data = results['raw_data']
    feature_data = results['feature_data']

    # Basic statistics
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Feature data shape: {feature_data.shape}")

    # Scenario distribution
    print("\nScenario distribution:")
    print(raw_data['scenario'].value_counts())

    # Fault events
    if 'rx_los' in raw_data.columns:
        rx_los_count = raw_data['rx_los'].sum()
        tx_fault_count = raw_data['tx_fault'].sum()
        print(f"\nFault events:")
        print(f"  Rx LOS: {rx_los_count}")
        print(f"  Tx Fault: {tx_fault_count}")

    # Feature statistics
    if not feature_data.empty:
        print(f"\nFeature columns: {list(feature_data.columns)}")
        print(f"Target variable distribution:")
        if 'target_rx_los_event_7d' in feature_data.columns:
            print(f"  target_rx_los_event_7d: {feature_data['target_rx_los_event_7d'].value_counts()}")
        if 'target_tx_fault_event_7d' in feature_data.columns:
            print(f"  target_tx_fault_event_7d: {feature_data['target_tx_fault_event_7d'].value_counts()}")


def visualize_data(results):
    """Create basic visualizations of the simulated data."""
    print("\n=== Creating Visualizations ===")

    raw_data = results['raw_data']

    # Set up plotting style
    plt.style.use('default')

    # Example: Plot metrics for first few modules
    sample_serials = raw_data['serial_number'].unique()[:3]

    for serial in sample_serials:
        module_data = raw_data[raw_data['serial_number'] == serial]
        scenario = module_data['scenario'].iloc[0]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Module {serial} - Scenario: {scenario}', fontsize=16)

        # Plot 1: Rx Power and Tx Power
        axes[0, 0].plot(module_data['timestamp'], module_data['rx_power'],
                       label='Rx Power', color='blue', alpha=0.7)
        axes[0, 0].plot(module_data['timestamp'], module_data['tx_power'],
                       label='Tx Power', color='red', alpha=0.7)
        axes[0, 0].set_ylabel('Power (dBm)')
        axes[0, 0].set_title('Optical Power')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Tx Bias Current
        axes[0, 1].plot(module_data['timestamp'], module_data['tx_bias'],
                       color='green', alpha=0.7)
        axes[0, 1].set_ylabel('Bias Current (mA)')
        axes[0, 1].set_title('Transmitter Bias Current')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Temperature
        axes[1, 0].plot(module_data['timestamp'], module_data['temperature'],
                       color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].set_title('Module Temperature')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: SNR
        axes[1, 1].plot(module_data['timestamp'], module_data['snr'],
                       color='purple', alpha=0.7)
        axes[1, 1].set_ylabel('SNR (dB)')
        axes[1, 1].set_title('Signal-to-Noise Ratio')
        axes[1, 1].grid(True, alpha=0.3)

        # Format x-axis
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'visualizations/module_{serial}_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization for module {serial}")


def demonstrate_fault_scenarios():
    """Demonstrate different fault scenarios."""
    print("\n=== Fault Scenario Demonstration ===")

    # Create a focused simulation with specific scenarios
    config = {
        'period_days': 14,      # 2 weeks
        'interval_minutes': 60, # 1-hour intervals
        'fault_ratio': 1.0,     # All modules have faults
        'num_modules': 4,       # 4 modules for demonstration
        'seed': 42
    }

    simulator = OpticalModuleSimulator(**config)

    # Override the random scenario assignment to show specific ones
    simulator.fault_scenarios = [
        'laser_aging',
        'fiber_contamination',
        'temperature_stress',
        'intermittent_fault'
    ]

    results = simulator.run_simulation()

    # Print scenario details
    raw_data = results['raw_data']
    for serial in raw_data['serial_number'].unique():
        module_data = raw_data[raw_data['serial_number'] == serial]
        scenario = module_data['scenario'].iloc[0]
        rx_los_events = module_data['rx_los'].sum()
        tx_fault_events = module_data['tx_fault'].sum()

        print(f"\nModule {serial}:")
        print(f"  Scenario: {scenario}")
        print(f"  Rx LOS events: {rx_los_events}")
        print(f"  Tx Fault events: {tx_fault_events}")


def main():
    """Run all examples and demonstrations."""

    print("Optical Module Fault Data Simulator - Examples")
    print("=" * 50)

    # Run basic example
    results1 = run_basic_example()

    # Run custom example
    results2 = run_custom_example()

    # Analyze data
    analyze_data(results1)

    # Create visualizations
    visualize_data(results1)

    # Demonstrate fault scenarios
    demonstrate_fault_scenarios()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("Generated files:")
    print("  - data/example_raw_data.csv")
    print("  - data/example_features.csv")
    print("  - metadata/example_metadata.json")
    print("  - visualizations/module_*.png (visualizations)")


if __name__ == "__main__":
    main()