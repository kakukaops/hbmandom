#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid


class OpticalModuleSimulator:
    """
    光模块故障数据仿真器。

    负责生成带时间序列特征的光模块监控数据，包含：
    - 正常工况下的噪声波动
    - 多种故障场景注入
    - 指标间的物理关联（如功率/SNR/FEC）
    - 多通道模块支持
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
        初始化仿真器参数与随机种子。

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
        """生成单个光模块的基础元数据。"""
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
        """为当前模块分配故障场景及场景参数。"""
        # 按故障比例决定是否进入故障场景，否则标记为健康样本。
        if random.random() < self.fault_ratio:
            scenario = random.choice(self.fault_scenarios[1:])
        else:
            scenario = "healthy"

        # 统一场景公共参数：故障起始时间和严重度。
        scenario_params = {
            "scenario": scenario,
            "fault_start_day": random.randint(
                int(self.period_days * 0.3), int(self.period_days * 0.8)
            ),
            "severity": random.uniform(0.5, 1.0),
        }

        # 仅为对应场景补充专属参数，避免无关字段污染后续逻辑。
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
        """生成单个模块在给定时间轴上的监控指标。"""

        n_samples = len(time_index)
        scenario = scenario_params["scenario"]
        fault_start_idx = int(
            scenario_params["fault_start_day"] * 24 * 60 / self.interval_minutes
        )

        df = pd.DataFrame(index=time_index)

        # 先生成健康基线数据，再根据场景注入故障扰动。
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

        # 初始故障标签来自阈值规则，后续可能被场景注入覆盖/增强。
        df["rx_los"] = (df["rx_power"] < -20).astype(int)
        df["tx_fault"] = ((df["tx_bias"] > 80) | (df["temperature"] > 85)).astype(int)
        df["rx_lol"] = ((df["snr"] < 12) & (df["rx_power"] > -25)).astype(int)

        if scenario != "healthy":
            # 非健康场景才执行故障注入，保证 healthy 数据保持纯净。
            self._apply_fault_scenario(df, scenario, scenario_params, fault_start_idx)

        df["serial_number"] = metadata["serial_number"]
        df["vendor"] = metadata["vendor"]
        df["model"] = metadata["model"]
        df["scenario"] = scenario

        return df

    def _apply_fault_scenario(
        self, df: pd.DataFrame, scenario: str, params: Dict, fault_start_idx: int
    ):
        """按场景将故障模式注入到指标序列。"""

        n_samples = len(df)
        severity = params["severity"]

        if scenario == "laser_aging":
            # 激光老化：先偏置电流抬升，超过阈值后带来发射功率衰减。
            aging_rate = params["aging_rate"] * severity
            power_decline_rate = params["power_decline_rate"] * severity

            for i in range(fault_start_idx, n_samples):
                days_since_fault = (
                    (i - fault_start_idx) * self.interval_minutes / (24 * 60)
                )

                bias_increase = aging_rate * days_since_fault
                df.iloc[i, df.columns.get_loc("tx_bias")] += bias_increase

                if bias_increase > 20:
                    # 偏置超出安全边界后，功率才进入退化阶段。
                    power_decline = power_decline_rate * (
                        days_since_fault - 20 / aging_rate
                    )
                    df.iloc[i, df.columns.get_loc("tx_power")] -= max(0, power_decline)

        elif scenario == "fiber_contamination":
            # 光纤污染：路径损耗随时间上升，SNR 同步下降。
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
            # 温度应力：温度逐步升高并受最大温升限制。
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
            # 突发故障：在固定时间窗内强制置为严重异常。
            failure_duration = int(24 * 60 / self.interval_minutes)
            end_idx = min(fault_start_idx + failure_duration, n_samples)

            df.iloc[fault_start_idx:end_idx, df.columns.get_loc("rx_power")] = -30
            df.iloc[fault_start_idx:end_idx, df.columns.get_loc("tx_power")] = -30
            df.iloc[fault_start_idx:end_idx, df.columns.get_loc("rx_los")] = 1

        elif scenario == "intermittent_fault":
            # 间歇故障：低概率随机触发，持续时间短且随机。
            for i in range(fault_start_idx, n_samples):
                if random.random() < 0.01:
                    duration = random.randint(1, int(60 / self.interval_minutes))
                    end_idx = min(i + duration, n_samples)

                    df.iloc[i:end_idx, df.columns.get_loc("rx_power")] -= (
                        random.uniform(5, 15)
                    )
                    df.iloc[i:end_idx, df.columns.get_loc("rx_los")] = 1

    def _calculate_snr(self, rx_power: pd.Series, temperature: pd.Series) -> pd.Series:
        """基于接收功率和温度估算 SNR。"""
        # 接收功率越高通常 SNR 越好，温度升高会拉低 SNR。
        base_snr = 30 + (rx_power - (-10)) * 1.5
        temp_effect = (temperature - 45) * -0.2
        noise = np.random.normal(0, 1, len(rx_power))

        snr = base_snr + temp_effect + noise
        return np.clip(snr, 0, 35)

    def _calculate_fec_errors(self, snr: pd.Series) -> pd.Series:
        """根据 SNR 生成 FEC 可纠错错误数。"""
        # SNR 越差，基础误码越高；再叠加突发分布模拟抖动。
        base_errors = 1000 * np.exp(-0.5 * snr)
        burst_errors = np.random.gamma(2, 2, len(snr))

        return np.round(base_errors * burst_errors)

    def run_simulation(self) -> Tuple[pd.DataFrame, Dict]:
        """执行完整仿真流程并返回原始数据与元数据。"""

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
            # 每 10 个模块输出一次进度，避免日志过于频繁。
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
        """打印仿真结果摘要。"""
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
