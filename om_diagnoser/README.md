# Optical Module Fault Data Simulator

A comprehensive Python-based simulator for generating realistic optical module time series data with various fault scenarios, designed for AI/ML-based fault prediction research and development.

## Overview

This simulator generates synthetic optical module monitoring data that mimics real-world behavior, including:
- **Normal operation** with realistic noise and daily cycles
- **Multiple fault scenarios** (laser aging, fiber contamination, temperature stress, etc.)
- **Physical relationships** between metrics (SNR, FEC errors, power levels)
- **Multi-lane support** for high-speed optical modules
- **Automatic feature engineering** for machine learning

## Features

### Core Simulation Capabilities
- **Configurable parameters**: period, sampling interval, fault ratio, number of modules
- **Realistic physics**: SNR calculations, FEC error modeling, temperature effects
- **Multiple fault scenarios**:
  - Laser aging (gradual bias current increase)
  - Fiber contamination (gradual path loss increase)
  - Temperature stress (overheating)
  - Sudden failure (instant link loss)
  - Intermittent faults (flapping behavior)

### Data Outputs
- **Raw time series data**: Physical metrics at regular intervals
- **Feature-engineered data**: ML-ready features with sliding window statistics
- **Metadata**: Module specifications, fault scenarios, and configuration
- **Visualizations**: Time series plots for analysis

### Machine Learning Support
- **Automatic feature generation**: Mean, standard deviation, trends, min/max values
- **Future event prediction**: Labels for future fault events
- **Event statistics**: Flap counts, time since last event
- **Normalized features**: Relative position in specification ranges

## Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib

### Quick Start
```bash
# Clone or download the simulator files
cd om_diagnoser

# Run the basic simulator
python optical_module_simulator.py

# Run examples
python example_usage.py
```

## Usage

### Basic Usage

```python
from optical_module_simulator import OpticalModuleSimulator

# Create simulator with default parameters
simulator = OpticalModuleSimulator(
    period_days=90,        # 90 days of data
    interval_minutes=15,   # 15-minute intervals
    fault_ratio=0.15,      # 15% of modules experience faults
    num_modules=50,        # 50 optical modules
    seed=42                # For reproducible results
)

# Run simulation
results = simulator.run_simulation()

# Export data
simulator.export_data(
    raw_output_path='my_raw_data.csv',
    feature_output_path='my_features.csv',
    metadata_output_path='my_metadata.json'
)
```

### Using Configuration Presets

```python
import simulator_config

# Use predefined configurations
config = simulator_config.STANDARD  # or QUICK_TEST, PRODUCTION, RESEARCH
simulator = OpticalModuleSimulator(**config)
```

### Custom Configuration

```python
# Custom parameters
custom_params = {
    'period_days': 30,
    'interval_minutes': 10,
    'fault_ratio': 0.2,
    'num_modules': 25,
    'seed': 123
}

simulator = OpticalModuleSimulator(**custom_params)
```

## Output Data Formats

### Raw Data Format
Each row represents a single timestamp for a specific module:

| Column | Description | Units |
|--------|-------------|--------|
| `timestamp` | Measurement time | datetime |
| `serial_number` | Module unique identifier | string |
| `vendor`, `model` | Module specifications | string |
| `rx_power` | Received optical power | dBm |
| `tx_power` | Transmitted optical power | dBm |
| `tx_bias` | Laser bias current | mA |
| `temperature` | Module temperature | °C |
| `snr` | Signal-to-noise ratio | dB |
| `fec_correctable` | FEC correctable errors | count |
| `rx_los` | Receiver loss of signal | 0/1 |
| `tx_fault` | Transmitter fault | 0/1 |
| `rx_lol` | Receiver loss of lock | 0/1 |
| `scenario` | Fault scenario | string |

### Feature Data Format
Each row represents a "health snapshot" for ML training:

| Feature Type | Examples | Description |
|-------------|----------|-------------|
| **Identifiers** | `snapshot_uuid`, `module_serial_number` | Unique identifiers |
| **Window Statistics** | `local_rx_power_mean_24h`, `local_tx_bias_stddev_24h` | 24-hour rolling statistics |
| **Trend Features** | `local_snr_trend_24h`, `local_temperature_trend_24h` | Rate of change indicators |
| **Event Statistics** | `rx_los_flap_count_24h`, `time_since_last_rx_los_hours` | Historical event patterns |
| **Relative Metrics** | `rx_power_relative_pos` | Position in specification range |
| **Prediction Targets** | `target_rx_los_event_7d`, `target_tx_fault_event_7d` | Future fault events |

## Fault Scenarios

### 1. Laser Aging
- **Symptoms**: Gradual increase in bias current, eventual power decline
- **Detection**: Rising `tx_bias_trend_24h`, declining `tx_power_trend_24h`
- **Physical cause**: Semiconductor aging in laser diode

### 2. Fiber Contamination
- **Symptoms**: Gradual increase in path loss, SNR degradation
- **Detection**: Rising `path_loss_trend_24h`, declining `snr_trend_24h`
- **Physical cause**: Dirty fiber connectors, contamination

### 3. Temperature Stress
- **Symptoms**: Elevated operating temperatures
- **Detection**: High `temperature_mean_24h`, positive `temperature_trend_24h`
- **Physical cause**: Poor cooling, environmental factors

### 4. Sudden Failure
- **Symptoms**: Instant loss of signal
- **Detection**: `rx_los` flag activation
- **Physical cause**: Physical damage, component failure

### 5. Intermittent Fault
- **Symptoms**: Random signal loss episodes
- **Detection**: High `rx_los_flap_count_24h`
- **Physical cause**: Loose connections, marginal components

## Machine Learning Applications

### Feature Engineering Strategy
The simulator automatically generates features suitable for:
- **Time series classification**: Predict future fault events
- **Anomaly detection**: Identify abnormal behavior patterns
- **Regression analysis**: Predict remaining useful life
- **Multi-label classification**: Predict multiple fault types

### Recommended ML Approaches
1. **XGBoost/LightGBM**: For tabular feature data
2. **LSTM/Transformer**: For raw time series data
3. **Isolation Forest**: For anomaly detection
4. **Survival Analysis**: For remaining useful life prediction

### Model Evaluation Metrics
- **Precision/Recall**: For imbalanced fault prediction
- **ROC-AUC**: For overall model performance
- **F1-Score**: Balanced metric for classification
- **Mean Time to Detection**: For early warning systems

## Configuration Options

### Simulation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `period_days` | 90 | Total simulation duration in days |
| `interval_minutes` | 15 | Time between samples in minutes |
| `fault_ratio` | 0.15 | Proportion of modules with faults |
| `num_modules` | 50 | Number of optical modules to simulate |
| `seed` | 42 | Random seed for reproducibility |

### Predefined Configurations
- **QUICK_TEST**: 7 days, 60-min intervals, 10 modules
- **STANDARD**: 90 days, 15-min intervals, 50 modules
- **PRODUCTION**: 365 days, 5-min intervals, 200 modules
- **RESEARCH**: 30 days, 1-min intervals, 20 modules

## Examples

See `example_usage.py` for comprehensive usage examples including:
- Basic simulation with different configurations
- Data analysis and visualization
- Fault scenario demonstrations
- Feature exploration

## File Structure

```
om_diagnoser/
├── optical_module_simulator.py  # Main simulator class
├── simulator_config.py          # Configuration presets
├── example_usage.py             # Usage examples
├── README.md                    # This documentation
└── generated_files/             # Output files (created during simulation)
    ├── *.csv                    # Raw and feature data
    ├── *.json                   # Metadata
    └── *.png                    # Visualizations
```

## Contributing

This simulator is designed to be extensible. Key extension points:

1. **New Fault Scenarios**: Add to `_apply_fault_scenario()` method
2. **Additional Metrics**: Extend `simulate_physical_metrics()`
3. **Custom Features**: Modify `generate_features()` method
4. **Export Formats**: Add new export methods

## License

This project is intended for research and educational purposes in optical network monitoring and AIOps development.

## References

Based on the comprehensive optical module monitoring and fault prediction requirements documented in `ompredict.md`, including:
- SFF-8472 Digital Diagnostic Monitoring (DDM)
- Optical module physical characteristics and failure modes
- Machine learning feature engineering for time series data
- Production monitoring system architectures