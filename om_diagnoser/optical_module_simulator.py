#!/usr/bin/env python3
"""
Optical Module Fault Data Simulator

This simulator generates realistic optical module time series data for fault prediction.
Based on the requirements from ompredict.md, it simulates various fault scenarios
including laser aging, fiber contamination, and sudden failures.

Author: liyan
Date: 2025-11-28
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid
import json


class OpticalModuleSimulator:
    """
    Optical Module Fault Data Simulator

    This class simulates optical module behavior over time, including:
    - Normal operation with realistic noise
    - Various fault scenarios (aging, contamination, sudden failures)
    - Physical relationships between metrics
    - Multi-lane support for high-speed modules
    """

    def __init__(self,
                 period_days: int = 90,
                 interval_minutes: int = 5,
                 fault_ratio: float = 0.1,
                 num_modules: int = 50,
                 seed: int = 42):
        """
        Initialize the simulator.

        Args:
            period_days: Total simulation period in days
            interval_minutes: Time interval between samples in minutes
            fault_ratio: Ratio of modules that will experience faults
            num_modules: Number of optical modules to simulate
            seed: Random seed for reproducibility
        """
        self.period_days = period_days
        self.interval_minutes = interval_minutes
        self.fault_ratio = fault_ratio
        self.num_modules = num_modules
        self.seed = seed

        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Calculate derived parameters
        self.total_samples = int(period_days * 24 * 60 / interval_minutes)
        self.sampling_freq = f'{interval_minutes}min'

        # Module specifications and baselines
        self.module_specs = {
            'rx_power_min': -14.0,  # dBm
            'rx_power_max': 2.0,    # dBm
            'tx_power_nominal': -2.0,  # dBm
            'tx_bias_nominal': 40.0,   # mA
            'temp_nominal': 45.0,      # °C
            'temp_max': 75.0,          # °C
            'voltage_nominal': 3.3,    # V
            'snr_nominal': 30.0,       # dB
        }

        # Fault scenarios
        self.fault_scenarios = [
            'healthy',
            'laser_aging',      # Gradual laser degradation
            'fiber_contamination',  # Fiber interface contamination
            'temperature_stress',   # Overheating
            'sudden_failure',       # Instant failure
            'intermittent_fault'    # Flapping behavior
        ]

        # Initialize data storage
        self.raw_data = []
        self.feature_data = []
        self.metadata = {}

    def generate_module_metadata(self) -> Dict:
        """Generate metadata for a single optical module."""
        vendors = ['Finisar', 'Cisco', 'Mellanox', 'Intel', 'Broadcom']
        models = ['QSFP-100G-SR4', 'QSFP28-100G-CWDM4', 'QSFP-DD-400G-DR4', 'SFP28-25G-SR']

        return {
            'serial_number': f"SN-{uuid.uuid4().hex[:8].upper()}",
            'vendor': random.choice(vendors),
            'model': random.choice(models),
            'num_lanes': random.choice([1, 4, 8]),
            'spec_rx_min': self.module_specs['rx_power_min'],
            'spec_rx_max': self.module_specs['rx_power_max'],
            'spec_temp_max': self.module_specs['temp_max'],
            'installation_date': datetime.now() - timedelta(days=random.randint(0, 365))
        }

    def assign_fault_scenario(self) -> Tuple[str, Dict]:
        """Assign a fault scenario to a module."""
        if random.random() < self.fault_ratio:
            scenario = random.choice(self.fault_scenarios[1:])  # Exclude 'healthy'
        else:
            scenario = 'healthy'

        # Scenario-specific parameters
        scenario_params = {
            'scenario': scenario,
            'fault_start_day': random.randint(int(self.period_days * 0.3), int(self.period_days * 0.8)),
            'severity': random.uniform(0.5, 1.0)
        }

        if scenario == 'laser_aging':
            scenario_params.update({
                'aging_rate': random.uniform(0.05, 0.2),  # mA/day
                'power_decline_rate': random.uniform(0.01, 0.05)  # dB/day
            })
        elif scenario == 'fiber_contamination':
            scenario_params.update({
                'contamination_rate': random.uniform(0.1, 0.5),  # dB/day
                'snr_decline_rate': random.uniform(0.2, 0.8)  # dB/day
            })
        elif scenario == 'temperature_stress':
            scenario_params.update({
                'temp_increase_rate': random.uniform(0.2, 1.0),  # °C/day
                'max_temp_offset': random.uniform(5, 15)  # °C above nominal
            })

        return scenario, scenario_params

    def simulate_physical_metrics(self,
                                metadata: Dict,
                                scenario_params: Dict,
                                time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Simulate physical metrics for a single module."""

        n_samples = len(time_index)
        scenario = scenario_params['scenario']
        fault_start_idx = int(scenario_params['fault_start_day'] * 24 * 60 / self.interval_minutes)

        # Initialize base metrics with realistic noise
        df = pd.DataFrame(index=time_index)

        # 1. Temperature with daily cycle
        base_temp = self.module_specs['temp_nominal']
        daily_cycle = 5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60 / self.interval_minutes))
        temp_noise = np.random.normal(0, 1.0, n_samples)
        df['temperature'] = base_temp + daily_cycle + temp_noise

        # 2. Voltage with minor fluctuations
        df['voltage'] = np.random.normal(self.module_specs['voltage_nominal'], 0.05, n_samples)

        # 3. Transmitter metrics
        df['tx_power'] = np.random.normal(self.module_specs['tx_power_nominal'], 0.1, n_samples)
        df['tx_bias'] = np.random.normal(self.module_specs['tx_bias_nominal'], 0.5, n_samples)

        # 4. Path loss (simulates fiber quality)
        df['path_loss'] = np.random.normal(3.0, 0.1, n_samples)

        # 5. Calculate derived metrics first
        # Receiver power = Transmitter power - Path loss (simplified)
        df['rx_power'] = df['tx_power'] - df['path_loss']

        # SNR based on physical relationships
        df['snr'] = self._calculate_snr(df['rx_power'], df['temperature'])

        # FEC errors based on SNR
        df['fec_correctable'] = self._calculate_fec_errors(df['snr'])

        # Status flags
        df['rx_los'] = (df['rx_power'] < -20).astype(int)
        df['tx_fault'] = ((df['tx_bias'] > 80) | (df['temperature'] > 85)).astype(int)
        df['rx_lol'] = ((df['snr'] < 12) & (df['rx_power'] > -25)).astype(int)

        # 6. Apply fault scenarios (now that all metrics are calculated)
        if scenario != 'healthy':
            self._apply_fault_scenario(df, scenario, scenario_params, fault_start_idx)

        # Add metadata
        df['serial_number'] = metadata['serial_number']
        df['vendor'] = metadata['vendor']
        df['model'] = metadata['model']
        df['scenario'] = scenario

        return df

    def _apply_fault_scenario(self,
                            df: pd.DataFrame,
                            scenario: str,
                            params: Dict,
                            fault_start_idx: int):
        """Apply specific fault scenario to the metrics."""

        n_samples = len(df)
        severity = params['severity']

        if scenario == 'laser_aging':
            # Gradual increase in bias current, eventual power decline
            aging_rate = params['aging_rate'] * severity
            power_decline_rate = params['power_decline_rate'] * severity

            for i in range(fault_start_idx, n_samples):
                days_since_fault = (i - fault_start_idx) * self.interval_minutes / (24 * 60)

                # Bias current increases linearly
                bias_increase = aging_rate * days_since_fault
                df.iloc[i, df.columns.get_loc('tx_bias')] += bias_increase

                # Power declines after bias exceeds threshold
                if bias_increase > 20:
                    power_decline = power_decline_rate * (days_since_fault - 20/aging_rate)
                    df.iloc[i, df.columns.get_loc('tx_power')] -= max(0, power_decline)

        elif scenario == 'fiber_contamination':
            # Gradual increase in path loss
            contamination_rate = params['contamination_rate'] * severity
            snr_decline_rate = params['snr_decline_rate'] * severity

            for i in range(fault_start_idx, n_samples):
                days_since_fault = (i - fault_start_idx) * self.interval_minutes / (24 * 60)

                # Path loss increases
                loss_increase = contamination_rate * days_since_fault
                df.iloc[i, df.columns.get_loc('path_loss')] += loss_increase

                # SNR declines
                snr_decline = snr_decline_rate * days_since_fault
                df.iloc[i, df.columns.get_loc('snr')] -= snr_decline

        elif scenario == 'temperature_stress':
            # Gradual temperature increase
            temp_increase_rate = params['temp_increase_rate'] * severity
            max_temp_offset = params['max_temp_offset']

            for i in range(fault_start_idx, n_samples):
                days_since_fault = (i - fault_start_idx) * self.interval_minutes / (24 * 60)

                # Temperature increases
                temp_increase = min(temp_increase_rate * days_since_fault, max_temp_offset)
                df.iloc[i, df.columns.get_loc('temperature')] += temp_increase

        elif scenario == 'sudden_failure':
            # Instant failure at fault start
            failure_duration = int(24 * 60 / self.interval_minutes)  # 24 hours of failure
            end_idx = min(fault_start_idx + failure_duration, n_samples)

            df.iloc[fault_start_idx:end_idx, df.columns.get_loc('rx_power')] = -30
            df.iloc[fault_start_idx:end_idx, df.columns.get_loc('tx_power')] = -30
            df.iloc[fault_start_idx:end_idx, df.columns.get_loc('rx_los')] = 1

        elif scenario == 'intermittent_fault':
            # Random flapping behavior
            for i in range(fault_start_idx, n_samples):
                if random.random() < 0.01:  # 1% chance of fault per sample
                    duration = random.randint(1, int(60 / self.interval_minutes))  # Up to 1 hour
                    end_idx = min(i + duration, n_samples)

                    df.iloc[i:end_idx, df.columns.get_loc('rx_power')] -= random.uniform(5, 15)
                    df.iloc[i:end_idx, df.columns.get_loc('rx_los')] = 1

    def _calculate_snr(self, rx_power: pd.Series, temperature: pd.Series) -> pd.Series:
        """Calculate Signal-to-Noise Ratio based on physical relationships."""
        # Base SNR model: higher rx_power and lower temperature -> better SNR
        base_snr = 30 + (rx_power - (-10)) * 1.5
        temp_effect = (temperature - 45) * -0.2  # Higher temp reduces SNR
        noise = np.random.normal(0, 1, len(rx_power))

        snr = base_snr + temp_effect + noise
        return np.clip(snr, 0, 35)

    def _calculate_fec_errors(self, snr: pd.Series) -> pd.Series:
        """Calculate FEC correctable errors based on SNR."""
        # Exponential model: FEC errors increase dramatically when SNR drops
        base_errors = 1000 * np.exp(-0.5 * snr)
        burst_errors = np.random.gamma(2, 2, len(snr))  # Random bursts

        return np.round(base_errors * burst_errors)

    def generate_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for machine learning from raw data."""

        features_list = []

        for serial_number, group in raw_data.groupby('serial_number'):
            df = group.copy()
            df = df.sort_values('timestamp')

            # Define window sizes
            window_size = int(24 * 60 / self.interval_minutes)  # 24 hours
            predict_window = int(7 * 24 * 60 / self.interval_minutes)  # 7 days

            # Initialize features DataFrame
            features = pd.DataFrame(index=df.index)

            # Basic identifiers
            features['snapshot_uuid'] = [uuid.uuid4().hex for _ in range(len(df))]
            features['snapshot_timestamp'] = df['timestamp']
            features['module_serial_number'] = df['serial_number']
            features['device_id'] = f"device_{df['serial_number'].iloc[0][-4:]}"
            features['vendor'] = df['vendor']
            features['model'] = df['model']

            # Continuous metrics for feature engineering
            metrics = ['rx_power', 'tx_power', 'tx_bias', 'temperature', 'snr', 'fec_correctable']

            # Calculate sliding window statistics
            for metric in metrics:
                if metric in df.columns:
                    # Mean
                    features[f'local_{metric}_mean_24h'] = df[metric].rolling(window=window_size).mean()
                    # Standard deviation
                    features[f'local_{metric}_stddev_24h'] = df[metric].rolling(window=window_size).std()
                    # Trend (simplified as difference from 24h ago)
                    features[f'local_{metric}_trend_24h'] = df[metric] - df[metric].shift(window_size)
                    # Minimum
                    features[f'local_{metric}_min_24h'] = df[metric].rolling(window=window_size).min()

            # Relative position in specification range
            features['rx_power_relative_pos'] = (
                (features['local_rx_power_mean_24h'] - self.module_specs['rx_power_min']) /
                (self.module_specs['rx_power_max'] - self.module_specs['rx_power_min'])
            )

            # Event statistics
            features['rx_los_flap_count_24h'] = (df['rx_los'].diff() == 1).rolling(window=window_size).sum()
            features['tx_fault_flap_count_24h'] = (df['tx_fault'].diff() == 1).rolling(window=window_size).sum()

            # Time since last critical event
            features['time_since_last_rx_los_hours'] = self._calculate_time_since_event(df, 'rx_los')
            features['time_since_last_tx_fault_hours'] = self._calculate_time_since_event(df, 'tx_fault')

            # Generate prediction targets (future events)
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=predict_window)

            features['target_rx_los_event_7d'] = df['rx_los'].rolling(window=indexer).max()
            features['target_tx_fault_event_7d'] = df['tx_fault'].rolling(window=indexer).max()
            features['target_fec_burst_7d'] = (
                df['fec_correctable'].rolling(window=indexer).max() > 1000
            ).astype(int)

            # Clean up NaN values (keep rows where we have enough data for features)
            # Skip the first window_size rows and last predict_window rows
            valid_start = window_size
            valid_end = len(features) - predict_window

            if valid_end > valid_start:
                features_clean = features.iloc[valid_start:valid_end]

                # Downsample for manageable dataset size
                if len(features_clean) > 100:
                    features_clean = features_clean.iloc[::4]  # Keep every 4th sample

                features_list.append(features_clean)

        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def _calculate_time_since_event(self, df: pd.DataFrame, event_col: str) -> pd.Series:
        """Calculate hours since last event occurrence."""
        time_since = pd.Series(index=df.index, dtype=float)
        last_event_idx = -1

        for i, (idx, row) in enumerate(df.iterrows()):
            if row[event_col] == 1:
                last_event_idx = i

            if last_event_idx >= 0:
                hours_since = (i - last_event_idx) * self.interval_minutes / 60
                time_since.iloc[i] = hours_since
            else:
                time_since.iloc[i] = np.nan

        return time_since

    def run_simulation(self) -> Dict:
        """Run the complete simulation and return results."""

        print(f"Starting optical module simulation...")
        print(f"Parameters: {self.period_days} days, {self.interval_minutes} min interval, "
              f"{self.fault_ratio} fault ratio, {self.num_modules} modules")

        # Generate time index
        start_time = datetime(2024, 1, 1)
        time_index = pd.date_range(
            start=start_time,
            periods=self.total_samples,
            freq=self.sampling_freq
        )

        all_raw_data = []
        all_metadata = {}

        # Simulate each module
        for i in range(self.num_modules):
            if i % 10 == 0:
                print(f"Simulating module {i+1}/{self.num_modules}...")

            # Generate module metadata
            metadata = self.generate_module_metadata()

            # Assign fault scenario
            scenario, scenario_params = self.assign_fault_scenario()

            # Store metadata
            all_metadata[metadata['serial_number']] = {
                'metadata': metadata,
                'scenario': scenario,
                'scenario_params': scenario_params
            }

            # Simulate physical metrics
            module_data = self.simulate_physical_metrics(metadata, scenario_params, time_index)
            module_data['timestamp'] = time_index

            all_raw_data.append(module_data)

        # Combine all data
        raw_df = pd.concat(all_raw_data, ignore_index=True)

        print("Generating features for machine learning...")
        feature_df = self.generate_features(raw_df)

        # Store results
        self.raw_data = raw_df
        self.feature_data = feature_df
        self.metadata = all_metadata

        # Print simulation summary
        self._print_summary()

        return {
            'raw_data': raw_df,
            'feature_data': feature_df,
            'metadata': all_metadata
        }

    def _print_summary(self):
        """Print summary of simulation results."""
        print("\n" + "="*50)
        print("SIMULATION SUMMARY")
        print("="*50)

        if self.raw_data is not None:
            print(f"Raw data shape: {self.raw_data.shape}")
            print(f"Feature data shape: {self.feature_data.shape}")

            # Count scenarios
            scenario_counts = self.raw_data['scenario'].value_counts()
            print("\nFault scenario distribution:")
            for scenario, count in scenario_counts.items():
                percentage = (count / len(self.raw_data)) * 100
                print(f"  {scenario}: {count} samples ({percentage:.1f}%)")

            # Count fault events
            if 'rx_los' in self.raw_data.columns:
                rx_los_count = self.raw_data['rx_los'].sum()
                tx_fault_count = self.raw_data['tx_fault'].sum()
                print(f"\nFault events:")
                print(f"  Rx LOS events: {rx_los_count}")
                print(f"  Tx Fault events: {tx_fault_count}")

        print("="*50)

    def export_data(self,
                   raw_output_path: str = 'data/optical_module_raw_data.csv',
                   feature_output_path: str = 'data/optical_module_features.csv',
                   metadata_output_path: str = 'metadata/optical_module_metadata.json'):
        """Export simulation data to files."""

        if self.raw_data is not None:
            self.raw_data.to_csv(raw_output_path, index=False)
            print(f"Raw data exported to: {raw_output_path}")

        if self.feature_data is not None:
            self.feature_data.to_csv(feature_output_path, index=False)
            print(f"Feature data exported to: {feature_output_path}")

        if self.metadata:
            # Convert datetime objects to strings for JSON serialization
            metadata_serializable = {}
            for sn, data in self.metadata.items():
                metadata_serializable[sn] = data.copy()
                if 'installation_date' in metadata_serializable[sn]['metadata']:
                    metadata_serializable[sn]['metadata']['installation_date'] = \
                        metadata_serializable[sn]['metadata']['installation_date'].isoformat()

            with open(metadata_output_path, 'w') as f:
                json.dump(metadata_serializable, f, indent=2)
            print(f"Metadata exported to: {metadata_output_path}")


def main():
    """Main function to demonstrate the simulator."""

    # Create simulator with example parameters
    simulator = OpticalModuleSimulator(
        period_days=60,        # 60 days of data
        interval_minutes=15,   # 15-minute intervals
        fault_ratio=0.15,      # 15% of modules will experience faults
        num_modules=30,        # 30 optical modules
        seed=42                # For reproducible results
    )

    # Run simulation
    results = simulator.run_simulation()

    # Export data
    simulator.export_data(
        raw_output_path='data/simulated_optical_module_data.csv',
        feature_output_path='data/optical_module_training_features.csv',
        metadata_output_path='metadata/optical_module_metadata.json'
    )

    print("\nSimulation completed successfully!")
    print("Generated files:")
    print("  - data/simulated_optical_module_data.csv (raw time series)")
    print("  - data/optical_module_training_features.csv (ML features)")
    print("  - metadata/optical_module_metadata.json (module information)")


if __name__ == "__main__":
    main()