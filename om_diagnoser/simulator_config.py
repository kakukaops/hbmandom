"""
Configuration file for Optical Module Fault Data Simulator

This file provides configuration presets for different simulation scenarios.
"""

# Quick test configuration (fast, small dataset)
QUICK_TEST = {
    'period_days': 7,           # 1 week
    'interval_minutes': 60,     # 1-hour intervals
    'fault_ratio': 0.1,         # 10% fault rate
    'num_modules': 10,          # 10 modules
    'seed': 42
}

# Standard configuration (balanced)
STANDARD = {
    'period_days': 90,          # 3 months
    'interval_minutes': 15,     # 15-minute intervals
    'fault_ratio': 0.15,        # 15% fault rate
    'num_modules': 50,          # 50 modules
    'seed': 42
}

# Production configuration (comprehensive)
PRODUCTION = {
    'period_days': 365,         # 1 year
    'interval_minutes': 5,      # 5-minute intervals
    'fault_ratio': 0.2,         # 20% fault rate
    'num_modules': 200,         # 200 modules
    'seed': 42
}

# Research configuration (high resolution)
RESEARCH = {
    'period_days': 30,          # 1 month
    'interval_minutes': 1,      # 1-minute intervals
    'fault_ratio': 0.25,        # 25% fault rate
    'num_modules': 20,          # 20 modules
    'seed': 42
}

# Custom configuration template
CUSTOM = {
    'period_days': 60,
    'interval_minutes': 30,
    'fault_ratio': 0.12,
    'num_modules': 25,
    'seed': 123
}

# Available configurations
CONFIGURATIONS = {
    'quick_test': QUICK_TEST,
    'standard': STANDARD,
    'production': PRODUCTION,
    'research': RESEARCH,
    'custom': CUSTOM
}