#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import uuid

from simulator import OpticalModuleSimulator


class OpticalModuleLogPreprocessor:
    """
    Optical Module Log Preprocessor

    This class preprocesses optical module log data for fault prediction.
    It supports both simulated data generation and real data processing.
    """

    def __init__(
        self,
        period_days: int = 90,
        fault_ratio: float = 0.1,
        num_modules: int = 50,
        seed: int = 42,
        with_simulation: bool = True,
        input_file: Optional[str] = None,
        config_path: Optional[str] = None,
        rules_path: Optional[str] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            period_days: Total simulation period in days
            fault_ratio: Ratio of modules that will experience faults
            num_modules: Number of optical modules to simulate
            seed: Random seed for reproducibility
            with_simulation: Whether to run simulation or read from input file
            input_file: Path to input CSV file (required if with_simulation=False)
            config_path: Path to info.yaml config file
            rules_path: Path to rules.yaml config file
        """
        self.period_days = period_days
        self.fault_ratio = fault_ratio
        self.num_modules = num_modules
        self.seed = seed
        self.with_simulation = with_simulation
        self.input_file = input_file

        self.config = self._load_yaml(config_path, "config/info.yaml")
        self.rules = self._load_yaml(rules_path, "config/rules.yaml")

        self.interval_minutes = self.config.get("interval_minutes", 15)
        self.predict_window_days = self.config.get("predict_window_days", 7)
        self.module_specs = self.config.get(
            "module_specs",
            {
                "rx_power_min": -14.0,
                "rx_power_max": 2.0,
                "tx_power_nominal": -2.0,
                "tx_bias_nominal": 40.0,
                "temp_nominal": 45.0,
                "temp_max": 75.0,
                "voltage_nominal": 3.3,
                "snr_nominal": 30.0,
            },
        )

        self.fault_types = self._get_fault_types()
        print(f"Loaded fault types from rules: {self.fault_types}")

        np.random.seed(seed)

        self.raw_data = []
        self.feature_data = []
        self.metadata = {}

    def _load_yaml(self, path: Optional[str], default_path: str) -> Dict:
        """Load configuration from YAML file."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), default_path)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file {path} not found, using defaults.")
            return {}

    def _get_fault_types(self) -> Set[str]:
        """Extract fault types (label_column) from rules.yaml."""
        fault_types = set()
        if self.rules and "rules" in self.rules:
            for rule in self.rules["rules"]:
                if "label_column" in rule:
                    fault_types.add(rule["label_column"])
        return fault_types

    def _get_target_column_name(self, fault_type: str) -> str:
        """Generate target column name dynamically based on fault type and predict window."""
        return f"target_{fault_type}_event_{self.predict_window_days}d"

    def generate_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for machine learning from raw data."""

        features_list = []

        for serial_number, group in raw_data.groupby("serial_number"):
            df = group.copy()
            df = df.sort_values("timestamp")

            window_size = int(24 * 60 / self.interval_minutes)
            predict_window = int(
                self.predict_window_days * 24 * 60 / self.interval_minutes
            )

            features = pd.DataFrame(index=df.index)

            features["snapshot_uuid"] = [uuid.uuid4().hex for _ in range(len(df))]
            features["snapshot_timestamp"] = df["timestamp"]
            features["module_serial_number"] = df["serial_number"]
            features["device_id"] = f"device_{df['serial_number'].iloc[0][-4:]}"
            features["vendor"] = df["vendor"]
            features["model"] = df["model"]

            metrics = [
                "rx_power",
                "tx_power",
                "tx_bias",
                "temperature",
                "snr",
                "fec_correctable",
            ]

            for metric in metrics:
                if metric in df.columns:
                    features[f"local_{metric}_mean_24h"] = (
                        df[metric].rolling(window=window_size).mean()
                    )
                    features[f"local_{metric}_stddev_24h"] = (
                        df[metric].rolling(window=window_size).std()
                    )
                    features[f"local_{metric}_trend_24h"] = df[metric] - df[
                        metric
                    ].shift(window_size)
                    features[f"local_{metric}_min_24h"] = (
                        df[metric].rolling(window=window_size).min()
                    )

            features["rx_power_relative_pos"] = (
                features["local_rx_power_mean_24h"] - self.module_specs["rx_power_min"]
            ) / (self.module_specs["rx_power_max"] - self.module_specs["rx_power_min"])

            indexer = pd.api.indexers.FixedForwardWindowIndexer(
                window_size=predict_window
            )

            for fault_type in self.fault_types:
                if fault_type in df.columns:
                    target_col = self._get_target_column_name(fault_type)
                    features[target_col] = df[fault_type].rolling(window=indexer).max()
                    if fault_type == "fec_burst":
                        features[target_col] = (
                            df["fec_correctable"].rolling(window=indexer).max() > 1000
                        ).astype(int)

            for fault_type in self.fault_types:
                if fault_type in df.columns:
                    features[f"{fault_type}_flap_count_24h"] = (
                        (df[fault_type].diff() == 1).rolling(window=window_size).sum()
                    )
                    features[f"time_since_last_{fault_type}_hours"] = (
                        self._calculate_time_since_event(df, fault_type)
                    )

            valid_start = window_size
            valid_end = len(features) - predict_window

            if valid_end > valid_start:
                features_clean = features.iloc[valid_start:valid_end]

                if len(features_clean) > 100:
                    features_clean = features_clean.iloc[::4]

                features_list.append(features_clean)

        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def _calculate_time_since_event(
        self, df: pd.DataFrame, event_col: str
    ) -> pd.Series:
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

    def run_preprocessing(self) -> Dict:
        """Run the complete preprocessing and return results."""

        print(f"Starting optical module log preprocessing...")
        print(
            f"Interval: {self.interval_minutes} min, Predict window: {self.predict_window_days} days"
        )
        print(f"Fault types: {self.fault_types}")

        if self.with_simulation:
            simulator = OpticalModuleSimulator(
                period_days=self.period_days,
                interval_minutes=self.interval_minutes,
                fault_ratio=self.fault_ratio,
                num_modules=self.num_modules,
                seed=self.seed,
                module_specs=self.module_specs,
                fault_scenarios=self.config.get("fault_scenarios"),
                vendors=self.config.get("vendors"),
                models=self.config.get("models"),
                num_lanes_options=self.config.get("num_lanes_options"),
            )
            raw_df, all_metadata = simulator.run_simulation()
        else:
            if self.input_file is None:
                raise ValueError("input_file is required when with_simulation=False")
            raw_df = pd.read_csv(self.input_file, parse_dates=["timestamp"])
            all_metadata = {}

        print("Generating features for machine learning...")
        feature_df = self.generate_features(raw_df)

        self.raw_data = raw_df
        self.feature_data = feature_df
        self.metadata = all_metadata

        self._print_summary()

        return {
            "raw_data": raw_df,
            "feature_data": feature_df,
            "metadata": all_metadata,
        }

    def _print_summary(self):
        """Print summary of preprocessing results."""
        print("\n" + "=" * 50)
        print("PREPROCESSING SUMMARY")
        print("=" * 50)

        if self.raw_data is not None and hasattr(self.raw_data, "shape"):
            print(f"Raw data shape: {self.raw_data.shape}")
        if self.feature_data is not None and hasattr(self.feature_data, "shape"):
            print(f"Feature data shape: {self.feature_data.shape}")

        if (
            self.with_simulation
            and hasattr(self.raw_data, "columns")
            and "scenario" in self.raw_data.columns
        ):
            scenario_counts = self.raw_data["scenario"].value_counts()
            print("\nFault scenario distribution:")
            for scenario, count in scenario_counts.items():
                percentage = (count / len(self.raw_data)) * 100
                print(f"  {scenario}: {count} samples ({percentage:.1f}%)")

            print(f"\nFault events by type:")
            for fault_type in self.fault_types:
                if fault_type in self.raw_data.columns:
                    count = self.raw_data[fault_type].sum()
                    print(f"  {fault_type}: {count} events")

        print("=" * 50)

    def export_data(
        self,
        raw_output_path: str = "optical_module_raw_data.csv",
        feature_output_path: str = "optical_module_features.csv",
        metadata_output_path: str = "optical_module_metadata.json",
    ):
        for data_dir in ["data", "metadata"]:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

        if (
            self.with_simulation
            and self.raw_data is not None
            and hasattr(self.raw_data, "to_csv")
        ):
            self.raw_data.to_csv(os.path.join("data", raw_output_path), index=False)
            print(f"Raw data exported to: {raw_output_path}")

        if self.feature_data is not None and hasattr(self.feature_data, "to_csv"):
            self.feature_data.to_csv(
                os.path.join("data", feature_output_path), index=False
            )
            print(f"Feature data exported to: {feature_output_path}")

        if self.metadata:
            metadata_serializable = {}
            for sn, data in self.metadata.items():
                metadata_serializable[sn] = data.copy()
                if (
                    "metadata" in metadata_serializable[sn]
                    and "installation_date" in metadata_serializable[sn]["metadata"]
                ):
                    metadata_serializable[sn]["metadata"]["installation_date"] = (
                        metadata_serializable[sn]["metadata"][
                            "installation_date"
                        ].isoformat()
                    )

            with open(os.path.join("data", metadata_output_path), "w") as f:
                json.dump(metadata_serializable, f, indent=2)
            print(f"Metadata exported to: {metadata_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Optical Module Log Preprocessor")
    parser.add_argument(
        "--period_days", type=int, default=60, help="Total simulation period in days"
    )
    parser.add_argument(
        "--fault_ratio",
        type=float,
        default=0.15,
        help="Ratio of modules that will experience faults",
    )
    parser.add_argument(
        "--num_modules",
        type=int,
        default=30,
        help="Number of optical modules to simulate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--raw_output_path",
        type=str,
        default="simulated_optical_module_data.csv",
        help="Output path for raw data CSV",
    )
    parser.add_argument(
        "--feature_output_path",
        type=str,
        default="optical_module_training_features.csv",
        help="Output path for feature data CSV",
    )
    parser.add_argument(
        "--metadata_output_path",
        type=str,
        default="optical_module_metadata.json",
        help="Output path for metadata JSON",
    )
    parser.add_argument(
        "--simulation", action="store_true", help="Run the optical module simulation"
    )
    parser.add_argument(
        "--input_file", type=str, help="Path to input CSV file for preprocessing"
    )
    parser.add_argument("--config", type=str, help="Path to info.yaml config file")
    parser.add_argument("--rules", type=str, help="Path to rules.yaml config file")
    args = parser.parse_args()

    preprocessor = OpticalModuleLogPreprocessor(
        period_days=args.period_days,
        fault_ratio=args.fault_ratio,
        num_modules=args.num_modules,
        seed=args.seed,
        with_simulation=args.simulation,
        input_file=args.input_file if not args.simulation else None,
        config_path=args.config,
        rules_path=args.rules,
    )

    preprocessor.run_preprocessing()

    preprocessor.export_data(
        raw_output_path=args.raw_output_path,
        feature_output_path=args.feature_output_path,
        metadata_output_path=args.metadata_output_path,
    )

    print("\nPreprocessing completed successfully!")
    print("Generated files:")
    print(f"  - data/{args.raw_output_path} (raw time series)")
    print(f"  - data/{args.feature_output_path} (ML features)")
    print(f"  - metadata/{args.metadata_output_path} (module information)")


if __name__ == "__main__":
    main()
