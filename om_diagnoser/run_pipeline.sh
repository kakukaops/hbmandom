#!/bin/bash
# 光模块故障预测一键运行脚本
# 执行数据标注 -> 特征生成 -> 模型训练完整流程

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_FILE="${1:-data/simulated_optical_module_data.csv}"
TARGET="${2:-rx_los}"

echo "=============================================="
echo "光模块故障预测一键运行脚本"
echo "=============================================="
echo "输入文件: $INPUT_FILE"
echo "预测目标: $TARGET"
echo ""

echo "[1/3] 运行数据标注..."
python auto_labeler.py

echo ""
echo "[2/3] 生成特征数据..."
python data_preprocessor.py --input_file data/labeled_optical_module_data.csv

echo ""
echo "[3/3] 训练预测模型..."
python om_fault_predictor.py --data data/optical_module_training_features.csv --target "$TARGET"

echo ""
echo "=============================================="
echo "流程执行完成!"
echo "=============================================="
echo "生成的文件:"
echo "  - data/labeled_optical_module_data.csv"
echo "  - data/optical_module_training_features.csv"
echo "  - models/om_fault_predictor.pkl"
echo "  - plots/model_evaluation.png"
echo "  - reports/model_evaluation_report.json"