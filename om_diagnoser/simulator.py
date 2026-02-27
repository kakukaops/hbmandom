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


class OpticalModuleSimulator:
    """
    Optical Module Fault Data Simulator

    This class simulates optical module behavior over time, including:
    - Normal operation with realistic noise
    - Various fault scenarios (aging, contamination, sudden failures)
    - Physical relationships between metrics
    - Multi-lane support for high-speed modules
    """

    def __init__(
        self,
        period_days: int = 90,
        interval_minutes: int = 5,
        fault_ratio: float = 0.1,
        num_modules: int = 50,
        seed: int = 42,
        module_specs: Optional[Dict] = None,
        fault_scenarios: Optional[List[str]] = None,
        vendors: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        num_lanes_options: Optional[List[int]] = None,
    ):
        """
        Initialize the simulator.

        Args:
            period_days: Total simulation period in days
            interval_minutes: Time interval between samples in minutes
            fault_ratio: Ratio of modules that will experience faults
            num_modules: Number of optical modules to simulate
            seed: Random seed for reproducibility
            module_specs: Module specifications and baselines
            fault_scenarios: List of fault scenarios
            vendors: List of vendor names
            models: List of model names
            num_lanes_options: List of lane count options
        """
        self.period_days = period_days
        self.interval_minutes = interval_minutes
        self.fault_ratio = fault_ratio
        self.num_modules = num_modules
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        self.total_samples = int(period_days * 24 * 60 / interval_minutes)
        self.sampling_freq = f"{interval_minutes}min"

        self.module_specs = module_specs or {
            "rx_power_min": -14.0,
            "rx_power_max": 2.0,
            "tx_power_nominal": -2.0,
            "tx_bias_nominal": 40.0,
            "temp_nominal": 45.0,
            "temp_max": 75.0,
            "voltage_nominal": 3.3,
            "snr_nominal": 30.0,
        }

        self.fault_scenarios = fault_scenarios or [
            "healthy",
            "laser_aging",
            "fiber_contamination",
            "temperature_stress",
            "sudden_failure",
            "intermittent_fault",
        ]

        self.vendors = vendors or ["Finisar", "Cisco", "Mellanox", "Intel", "Broadcom"]
        self.models = models or [
            "QSFP-100G-SR4",
            "QSFP28-100G-CWDM4",
            "QSFP-DD-400G-DR4",
            "SFP28-25G-SR",
        ]
        self.num_lanes_options = num_lanes_options or [1, 4, 8]

        self.raw_data = []
        self.metadata = {}

    def generate_module_metadata(self) -> Dict:
        """Generate metadata for a single optical module."""
        return {
            "serial_number": f"SN-{uuid.uuid4().hex[:8].upper()}",
            "vendor": random.choice(self.vendors),
            "model": random.choice(self.models),
            "num_lanes": random.choice(self.num_lanes_options),
            "spec_rx_min": self.module_specs["rx_power_min"],
            "spec_rx_max": self.module_specs["rx_power_max"],
            "spec_temp_max": self.module_specs["temp_max"],
            "installation_date": datetime.now()
            - timedelta(days=random.randint(0, 365)),
        }

    def assign_fault_scenario(self) -> Tuple[str, Dict]:
        """Assign a fault scenario to a module."""
        if random.random() < self.fault_ratio:
            scenario = random.choice(self.fault_scenarios[1:])
        else:
            scenario = "healthy"

        scenario_params = {
            "scenario": scenario,
            "fault_start_day": random.randint(
                int(self.period_days * 0.3), int(self.period_days * 0.8)
            ),
            "severity": random.uniform(0.5, 1.0),
        }

        if scenario == "laser_aging":
            scenario_params.update(
                {
                    "aging_rate": random.uniform(0.05, 0.2),
                    "power_decline_rate": random.uniform(0.01, 0.05),
                }
            )
        elif scenario == "fiber_contamination":
            scenario_params.update(
                {
                    "contamination_rate": random.uniform(0.1, 0.5),
                    "snr_decline_rate": random.uniform(0.2, 0.8),
                }
            )
        elif scenario == "temperature_stress":
            scenario_params.update(
                {
                    "temp_increase_rate": random.uniform(0.2, 1.0),
                    "max_temp_offset": random.uniform(5, 15),
                }
            )

        return scenario, scenario_params

    def simulate_physical_metrics(
        self, metadata: Dict, scenario_params: Dict, time_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Simulate physical metrics for a single module."""

        n_samples = len(time_index)
        scenario = scenario_params["scenario"]
        fault_start_idx = int(
            scenario_params["fault_start_day"] * 24 * 60 / self.interval_minutes
        )

        df = pd.DataFrame(index=time_index)

        base_temp = self.module_specs["temp_nominal"]
        daily_cycle = 5 * np.sin(
            2 * np.pi * np.arange(n_samples) / (24 * 60 / self.interval_minutes)
        )
        temp_noise = np.random.normal(0, 1.0, n_samples)
        df["temperature"] = base_temp + daily_cycle + temp_noise

        df["voltage"] = np.random.normal(
            self.module_specs["voltage_nominal"], 0.05, n_samples
        )

        df["tx_power"] = np.random.normal(
            self.module_specs["tx_power_nominal"], 0.1, n_samples
        )
        df["tx_bias"] = np.random.normal(
            self.module_specs["tx_bias_nominal"], 0.5, n_samples
        )

        df["path_loss"] = np.random.normal(3.0, 0.1, n_samples)

        df["rx_power"] = df["tx_power"] - df["path_loss"]

        df["snr"] = self._calculate_snr(df["rx_power"], df["temperature"])

        df["fec_correctable"] = self._calculate_fec_errors(df["snr"])

        df["rx_los"] = (df["rx_power"] < -20).astype(int)
        df["tx_fault"] = ((df["tx_bias"] > 80) | (df["temperature"] > 85)).astype(int)
        df["rx_lol"] = ((df["snr"] < 12) & (df["rx_power"] > -25)).astype(int)

        if scenario != "healthy":
            self._apply_fault_scenario(df, scenario, scenario_params, fault_start_idx)

        df["serial_number"] = metadata["serial_number"]
        df["vendor"] = metadata["vendor"]
        df["model"] = metadata["model"]
        df["scenario"] = scenario

        return df

    def _apply_fault_scenario(
        self, df: pd.DataFrame, scenario: str, params: Dict, fault_start_idx: int
    ):
        """Apply specific fault scenario to the metrics."""

        n_samples = len(df)
        severity = params["severity"]

        if scenario == "laser_aging":
            aging_rate = params["aging_rate"] * severity
            power_decline_rate = params["power_decline_rate"] * severity

            for i in range(fault_start_idx, n_samples):
                days_since_fault = (
                    (i - fault_start_idx) * self.interval_minutes / (24 * 60)
                )

                bias_increase = aging_rate * days_since_fault
                df.iloc[i, df.columns.get_loc("tx_bias")] += bias_increase

                if bias_increase > 20:
                    power_decline = power_decline_rate * (
                        days_since_fault - 20 / aging_rate
                    )
                    df.iloc[i, df.columns.get_loc("tx_power")] -= max(0, power_decline)

        elif scenario == "fiber_contamination":
            contamination_rate = params["contamination_rate"] * severity
            snr_decline_rate = params["snr_decline_rate"] * severity

            for i in range(fault_start_idx, n_samples):
                days_since_fault = (
                    (i - fault_start_idx) * self.interval_minutes / (24 * 60)
                )

                loss_increase = contamination_rate * days_since_fault
                df.iloc[i, df.columns.get_loc("path_loss")] += loss_increase

                snr_decline = snr_decline_rate * days_since_fault
                df.iloc[i, df.columns.get_loc("snr")] -= snr_decline

        elif scenario == "temperature_stress":
            temp_increase_rate = params["temp_increase_rate"] * severity
            max_temp_offset = params["max_temp_offset"]

            for i in range(fault_start_idx, n_samples):
                days_since_fault = (
                    (i - fault_start_idx) * self.interval_minutes / (24 * 60)
                )

                temp_increase = min(
                    temp_increase_rate * days_since_fault, max_temp_offset
                )
                df.iloc[i, df.columns.get_loc("temperature")] += temp_increase

        elif scenario == "sudden_failure":
            failure_duration = int(24 * 60 / self.interval_minutes)
            end_idx = min(fault_start_idx + failure_duration, n_samples)

            df.iloc[fault_start_idx:end_idx, df.columns.get_loc("rx_power")] = -30
            df.iloc[fault_start_idx:end_idx, df.columns.get_loc("tx_power")] = -30
            df.iloc[fault_start_idx:end_idx, df.columns.get_loc("rx_los")] = 1

        elif scenario == "intermittent_fault":
            for i in range(fault_start_idx, n_samples):
                if random.random() < 0.01:
                    duration = random.randint(1, int(60 / self.interval_minutes))
                    end_idx = min(i + duration, n_samples)

                    df.iloc[i:end_idx, df.columns.get_loc("rx_power")] -= (
                        random.uniform(5, 15)
                    )
                    df.iloc[i:end_idx, df.columns.get_loc("rx_los")] = 1

    def _calculate_snr(self, rx_power: pd.Series, temperature: pd.Series) -> pd.Series:
        """Calculate Signal-to-Noise Ratio based on physical relationships."""
        base_snr = 30 + (rx_power - (-10)) * 1.5
        temp_effect = (temperature - 45) * -0.2
        noise = np.random.normal(0, 1, len(rx_power))

        snr = base_snr + temp_effect + noise
        return np.clip(snr, 0, 35)

    def _calculate_fec_errors(self, snr: pd.Series) -> pd.Series:
        """Calculate FEC correctable errors based on SNR."""
        base_errors = 1000 * np.exp(-0.5 * snr)
        burst_errors = np.random.gamma(2, 2, len(snr))

        return np.round(base_errors * burst_errors)

    def run_simulation(self) -> Tuple[pd.DataFrame, Dict]:
        """Run the complete simulation and return results."""

        print(f"Starting optical module simulation...")
        print(
            f"Parameters: {self.period_days} days, {self.interval_minutes} min interval, "
            f"{self.fault_ratio} fault ratio, {self.num_modules} modules"
        )

        start_time = datetime(2024, 1, 1)
        time_index = pd.date_range(
            start=start_time, periods=self.total_samples, freq=self.sampling_freq
        )

        all_raw_data = []
        all_metadata = {}

        for i in range(self.num_modules):
            if i % 10 == 0:
                print(f"Simulating module {i + 1}/{self.num_modules}...")

            metadata = self.generate_module_metadata()

            scenario, scenario_params = self.assign_fault_scenario()

            all_metadata[metadata["serial_number"]] = {
                "metadata": metadata,
                "scenario": scenario,
                "scenario_params": scenario_params,
            }

            module_data = self.simulate_physical_metrics(
                metadata, scenario_params, time_index
            )
            module_data["timestamp"] = time_index

            all_raw_data.append(module_data)

        raw_df = pd.concat(all_raw_data, ignore_index=True)

        self.raw_data = raw_df
        self.metadata = all_metadata

        self._print_summary()

        return raw_df, all_metadata

    def _print_summary(self):
        """Print summary of simulation results."""
        print("\n" + "=" * 50)
        print("SIMULATION SUMMARY")
        print("=" * 50)

        if self.raw_data is not None:
            print(f"Raw data shape: {self.raw_data.shape}")

            scenario_counts = self.raw_data["scenario"].value_counts()
            print("\nFault scenario distribution:")
            for scenario, count in scenario_counts.items():
                percentage = (count / len(self.raw_data)) * 100
                print(f"  {scenario}: {count} samples ({percentage:.1f}%)")

            if "rx_los" in self.raw_data.columns:
                rx_los_count = self.raw_data["rx_los"].sum()
                tx_fault_count = self.raw_data["tx_fault"].sum()
                print(f"\nFault events:")
                print(f"  Rx LOS events: {rx_los_count}")
                print(f"  Tx Fault events: {tx_fault_count}")

        print("=" * 50)
