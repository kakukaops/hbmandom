# 光模块故障数据仿真与预测

光模块故障预测算法，使用仿真数据训练模型，能够预测光模块在未来7天内发生接收信号丢失(Rx LOS)故障的概率。

## 项目结构

```
om_diagnoser/
├── optical_module_simulator.py    # 光模块故障数据仿真与特征抽取
├── om_fault_predictor.py          # XGBoost故障预测模型训练
├── predict_faults.py              # 故障预测脚本
├── README.md                      # 项目说明文档
│
├── data/                          # 数据目录
│   ├── simulated_optical_module_data.csv      # 原始仿真数据
│   └── optical_module_training_features.csv   # 特征工程后的训练数据
│
├── models/                        # 模型文件目录
│   ├── om_fault_predictor.pkl              # 训练好的XGBoost模型
│   ├── om_fault_predictor_scaler.pkl       # 特征标准化器
│   ├── om_fault_predictor_encoders.pkl     # 分类变量编码器
│   ├── om_fault_predictor_features.json    # 特征名称
│   └── om_fault_predictor_metadata.json    # 模型元数据
│
├── reports/                       # 评估报告
│   └── model_evaluation_report.json        # 模型评估结果
│
├── plots/                         # 可视化图表
│   └── model_evaluation.png                # 模型评估图表
│
└── predictions/                   # 预测结果
    └── test_predictions.csv                # 测试预测结果
```

## 数据仿真与模型训练

### 1. 环境要求
```bash
uv pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### 2. 生成仿真数据

```bash
python optical_module_simulator.py
```

仿真器支持5种故障场景：

1. **激光器老化** - 偏置电流逐渐增加，最终导致功率下降
2. **光纤污染** - 路径损耗逐渐增加，SNR下降
3. **温度应力** - 温度逐渐升高超过额定值
4. **突发故障** - 瞬时完全故障，持续24小时
5. **间歇性故障** - 随机发生的瞬时故障

生成：
- `data/simulated_optical_module_data.csv` - 原始时间序列数据
- `data/optical_module_training_features.csv` - 处理后的特征数据
- `metadata/optical_module_metadata.json` - 仿真生成的光模块元数据

### 3. 训练预测模型
```bash
python om_fault_predictor.py
```
这将：
- 加载特征数据
- 训练XGBoost模型
- 评估模型性能
- 保存模型到`models/`目录
- 生成可视化图表到`plots/`

### 4. 使用模型进行预测

#### 批量预测：
```bash
python predict_faults.py --batch data/new_data.csv --output predictions/results.csv
```

#### 单样本预测（Python代码）：
```python
from predict_faults import FaultPredictor

# 初始化预测器
predictor = FaultPredictor()

# 准备特征数据
features = {
    'vendor': 'Cisco',
    'model': 'QSFP28-100G-CWDM4',
    # ... 其他特征
}

# 进行预测
result = predictor.predict_single(features)
print(f"预测结果: {result}")
```

#### 运行示例：
```bash
python predict_faults.py --example
```