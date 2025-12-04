# Optical Module Fault Prediction System

基于XGBoost的光模块故障预测系统，使用仿真数据训练模型，能够预测光模块在未来7天内发生接收信号丢失(Rx LOS)故障的概率。

## 项目结构

```
om_diagnoser/
├── optical_module_simulator.py    # 光模块故障数据仿真器
├── om_fault_predictor.py          # XGBoost故障预测模型训练
├── predict_faults.py              # 故障预测脚本
├── test_prediction.py             # 预测功能测试
├── README_FAULT_PREDICTION.md     # 项目说明文档
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

## 功能特性

### 1. 数据仿真 (`optical_module_simulator.py`)
- 模拟多种光模块故障场景：
  - 激光器老化 (laser_aging)
  - 光纤污染 (fiber_contamination)
  - 温度应力 (temperature_stress)
  - 突发故障 (sudden_failure)
  - 间歇性故障 (intermittent_fault)
- 生成包含物理指标的时间序列数据
- 自动生成机器学习特征

### 2. 故障预测模型 (`om_fault_predictor.py`)
- 基于XGBoost的二分类模型
- 预测未来7天内发生Rx LOS故障的概率
- 支持交叉验证和超参数调整
- 自动特征工程和数据预处理

### 3. 预测功能 (`predict_faults.py`)
- 加载训练好的模型进行实时预测
- 支持批量预测和单样本预测
- 可调节预测阈值
- 输出风险等级（低/中/高）

## 模型性能

基于仿真数据的模型表现优异：

| 指标 | 数值 | 说明 |
|------|------|------|
| 准确率 | 99.53% | 整体预测准确率 |
| 精确率 | 90.84% | 正类预测的准确率 |
| 召回率 | 100.00% | 正类样本的检出率 |
| F1分数 | 95.20% | 精确率和召回率的调和平均 |
| ROC-AUC | 99.98% | 模型区分能力 |

**混淆矩阵：**
- 真阴性(TN): 7106
- 假阳性(FP): 35
- 假阴性(FN): 0
- 真阳性(TP): 347

## 快速开始

### 1. 环境要求
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### 2. 生成仿真数据
```bash
python optical_module_simulator.py
```
这将生成：
- `data/simulated_optical_module_data.csv` - 原始时间序列数据
- `data/optical_module_training_features.csv` - 机器学习特征数据
- `metadata/optical_module_metadata.json` - 模块元数据

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
python test_prediction.py
```

## 特征重要性

模型识别出的关键预测特征：

1. **time_since_last_rx_los_hours** (25.2%) - 距离上次Rx LOS的小时数
2. **rx_los_flap_count_24h** (14.9%) - 24小时内Rx LOS翻转次数
3. **model** (14.2%) - 光模块型号
4. **vendor** (9.6%) - 供应商
5. **device_id** (6.7%) - 设备ID
6. **local_snr_stddev_24h** (3.0%) - SNR的24小时标准差
7. **local_snr_min_24h** (2.7%) - SNR的24小时最小值
8. **local_snr_mean_24h** (2.1%) - SNR的24小时均值
9. **local_rx_power_min_24h** (2.0%) - 接收功率的24小时最小值
10. **rx_power_relative_pos** (2.0%) - 接收功率在规格范围内的相对位置

## 故障场景模拟

仿真器支持5种故障场景：

1. **激光器老化** - 偏置电流逐渐增加，最终导致功率下降
2. **光纤污染** - 路径损耗逐渐增加，SNR下降
3. **温度应力** - 温度逐渐升高超过额定值
4. **突发故障** - 瞬时完全故障，持续24小时
5. **间歇性故障** - 随机发生的瞬时故障

## 扩展和改进建议

### 1. 数据增强
- 增加更多故障场景的仿真数据
- 调整故障比例以获得更平衡的数据集
- 添加更多物理指标和派生特征

### 2. 模型优化
- 使用网格搜索进行超参数调优
- 尝试其他算法（LightGBM, CatBoost, 神经网络）
- 实现多目标预测（同时预测多种故障类型）

### 3. 部署功能
- 创建REST API服务
- 添加实时数据流处理
- 实现模型版本管理和A/B测试
- 添加监控和告警功能

### 4. 生产环境考虑
- 添加数据验证和异常检测
- 实现模型漂移检测和再训练
- 添加预测解释功能（SHAP值）
- 创建仪表板和报告系统

## 技术细节

### 数据预处理流程
1. 缺失值处理：时间相关特征用大值填充（表示无事件）
2. 分类变量编码：使用LabelEncoder
3. 特征标准化：使用StandardScaler
4. 特征选择：基于ANOVA F-value选择重要特征

### 模型配置
- 算法：XGBoost
- 目标函数：binary:logistic
- 评估指标：AUC
- 树深度：6
- 学习率：0.1
- 树数量：100
- 正类权重调整：处理类别不平衡

## 故障预测业务价值

1. **预防性维护** - 提前预测故障，减少非计划停机
2. **资源优化** - 合理安排维护计划，降低运营成本
3. **风险管控** - 识别高风险设备，优先处理
4. **库存管理** - 优化备件库存，减少资金占用
5. **服务质量** - 提高网络可靠性，提升客户满意度

## 许可证

本项目仅供学习和研究使用。