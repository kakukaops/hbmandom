#!/bin/bash
# 光模块故障预测一键运行脚本
# 执行数据标注 -> 特征生成 -> 模型训练完整流程
# 所有故障类型均从 config/rules.yaml 中动态获取

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_FILE="${1:-data/simulated_optical_module_data.csv}"
TARGET="${2:-}"

echo "=============================================="
echo "光模块故障预测一键运行脚本"
echo "如果没有测试数据，请运行 python data_preprocessor.py --simulation 来生成模拟数据"
echo "=============================================="
echo "输入文件: $INPUT_FILE"
echo "故障类型: 从 config/rules.yaml 动态获取"
if [ -n "$TARGET" ]; then
    echo "指定目标: $TARGET"
fi
echo ""

echo "[1/3] 运行数据标注..."
python auto_labeler.py

echo ""
echo "[2/3] 生成特征数据..."
python data_preprocessor.py --input_file data/labeled_optical_module_data.csv

echo ""
echo "[3/3] 训练预测模型..."
if [ -n "$TARGET" ]; then
    python om_fault_predictor.py --data data/optical_module_training_features.csv --target "$TARGET"
else
    python om_fault_predictor.py --data data/optical_module_training_features.csv
fi
