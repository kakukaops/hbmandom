#!/bin/bash
# 光模块故障预测一键运行脚本
# 执行数据标注 -> 特征生成 -> 模型训练完整流程
# 所有故障类型均从 config/rules.yaml 中动态获取

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_FILE="${1:-data/simulated_optical_module_data.csv}"
TARGET="${2:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "错误: 未找到 python/python3 解释器"
        exit 1
    fi
fi

echo "=============================================="
echo "光模块故障预测一键运行脚本"
echo "如果没有测试数据，请运行 $PYTHON_BIN data_preprocessor.py --simulation 来生成模拟数据"
echo "=============================================="
echo "输入文件: $INPUT_FILE"
echo "故障类型: 从 config/rules.yaml 动态获取"
# 仅在用户显式传入目标时打印，便于区分默认行为和指定行为。
if [ -n "$TARGET" ]; then
    echo "指定目标: $TARGET"
fi
echo ""

if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    echo "请先生成数据，或传入正确的输入文件路径"
    exit 1
fi

echo "[1/3] 运行数据标注..."
"$PYTHON_BIN" auto_labeler.py \
    --input_path "$INPUT_FILE" \
    --output_path data/labeled_optical_module_data.csv

echo ""
echo "[2/3] 生成特征数据..."
"$PYTHON_BIN" data_preprocessor.py --input_file data/labeled_optical_module_data.csv

echo ""
echo "[3/3] 训练预测模型..."
# 若提供 TARGET，则训练指定故障类型；否则使用训练脚本默认目标。
if [ -n "$TARGET" ]; then
    "$PYTHON_BIN" om_fault_predictor.py --data data/optical_module_training_features.csv --target "$TARGET"
else
    "$PYTHON_BIN" om_fault_predictor.py --data data/optical_module_training_features.csv
fi
