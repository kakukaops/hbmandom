# 光模块故障数据仿真与预测

光模块故障预测算法，使用仿真数据训练模型，能够预测光模块在未来7天内发生故障的概率。

## 环境要求

- **Python**: >= 3.8
- **依赖包**: 见 `requirements.txt`

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 uv
uv pip install -r requirements.txt
```

## 项目结构

```
om_diagnoser/
├── simulator.py                    # 光模块故障数据仿真器
├── data_preprocessor.py            # 数据预处理与特征抽取
├── auto_labeler.py                 # 自动特征标注
├── om_fault_predictor.py           # XGBoost故障预测模型训练
├── predict_faults.py               # 故障预测脚本
├── simulator_config.py             # 仿真器预设配置
├── run_pipeline.sh                 # 一键运行脚本
├── requirements.txt                # Python依赖包
├── README.md                       # 项目说明文档
│
├── config/                         # 配置文件目录
│   ├── info.yaml                   # 模块规格、时间间隔、预测窗口等配置
│   ├── rules.yaml                  # 标注规则配置
│   └── hyper_parameters.yaml       # XGBoost超参数配置
│
├── data/                           # 数据目录
│   ├── simulated_optical_module_data.csv      # 原始仿真数据
│   ├── labeled_optical_module_data.csv        # 标注后的特征数据
│   ├── labeling_stats.json                    # 标注信息元数据
│   └── optical_module_training_features.csv   # 特征工程后的训练数据
│
├── example_data/                   # 示例数据目录
│   ├── features_sample.csv         # 特征数据样本
│   ├── prediction_sample.csv       # 预测结果样本
│   └── models/                     # 示例模型
│       └── om_fault_predictor.pkl  # 预训练模型
│
├── metadata/                       # 元数据目录
│   └── optical_module_metadata.json           # 光模块元数据
│
├── models/                         # 模型文件目录
│   ├── om_fault_predictor.pkl              # 训练好的XGBoost模型
│   ├── om_fault_predictor_scaler.pkl       # 特征标准化器
│   ├── om_fault_predictor_encoders.pkl     # 分类变量编码器
│   ├── om_fault_predictor_features.json    # 特征名称
│   └── om_fault_predictor_metadata.json    # 模型元数据
│
├── reports/                        # 评估报告
│   └── model_evaluation_report.json        # 模型评估结果
│
├── plots/                          # 可视化图表
│   └── model_evaluation.png                # 模型评估图表
│
└── predictions/                    # 预测结果
    └── test_predictions.csv                # 测试预测结果
```

## 快速开始

### 一键运行

使用示例数据一键运行完整流程（数据标注 → 特征生成 → 模型训练）：

```bash
# 使用默认参数（输入：data/simulated_optical_module_data.csv，目标：rx_los）
./run_pipeline.sh

# 指定输入文件和预测目标
./run_pipeline.sh data/your_data.csv tx_fault
```

支持的预测目标（需在 `config/rules.yaml` 中定义）：
- `rx_los` - 接收端信号丢失
- `tx_fault` - 发送端故障
- `rx_lol` - 接收端失锁
- `fec_burst` - FEC突发错误

## 数据仿真与模型训练

### 1. 数据处理

#### 1.1 配置说明

| 配置文件 | 说明 |
|---------|------|
| `config/info.yaml` | 模块规格参数、采样间隔、预测窗口 |
| `config/rules.yaml` | 标注规则定义（输入输出路径、条件规则） |
| `config/hyper_parameters.yaml` | XGBoost超参数配置 |

`config/info.yaml` 主要配置项：
- `module_specs`: 光模块规格参数（接收功率范围、发射功率标称值等）
- `interval_minutes`: 数据采样间隔（分钟）
- `predict_window_days`: 预测窗口（天数）

`config/hyper_parameters.yaml` 主要配置项：
- `xgboost.max_depth`: 树的最大深度
- `xgboost.learning_rate`: 学习率
- `xgboost.n_estimators`: 树的数量
- `data_split.test_size`: 测试集比例
- `cross_validation.n_splits`: 交叉验证折数

#### 1.2 生成仿真数据

```bash
# 生成仿真数据并抽取特征
python data_preprocessor.py --simulation --period_days 30 --num_modules 10 --fault_ratio 0.2

# 参数说明：
#   --period_days      仿真周期（天）
#   --num_modules      仿真光模块数量
#   --fault_ratio      故障比例
#   --seed             随机种子
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

#### 1.3 标注数据

支持基于自定义告警规则标注数据：

```bash
python auto_labeler.py
```

1. 在 `config/rules.yaml` 中配置输入输出文件路径、标注规则
2. 支持自定义操作符、自定义标注label

标注后生成：
- `data/labeled_optical_module_data.csv` - 标注后的数据

#### 1.4 基于已有数据抽取特征

```bash
# 基于收集的指标数据或标注后的数据抽取特征
python data_preprocessor.py --input_file data/labeled_optical_module_data.csv
```

### 2. 训练预测模型

```bash
# 使用默认配置训练 rx_los 预测模型
python om_fault_predictor.py

# 指定预测目标
python om_fault_predictor.py --target tx_fault

# 指定数据文件和超参数配置
python om_fault_predictor.py --data data/features.csv --target rx_los --hyperparams config/hyper_parameters.yaml
```

命令行参数：
| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--data` | 特征数据文件路径 | `data/optical_module_training_features.csv` |
| `--target` | 预测目标（rx_los/tx_fault/rx_lol/fec_burst） | `rx_los` |
| `--hyperparams` | 超参数配置文件路径 | `config/hyper_parameters.yaml` |
| `--rules` | 标注规则配置文件路径 | `config/rules.yaml` |

训练完成后将：
- 加载特征数据
- 训练XGBoost模型
- 评估模型性能
- 保存模型到`models/`目录
- 生成可视化图表到`plots/`

### 3. 使用模型进行预测

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

## 示例数据

`example_data/` 目录包含示例数据，方便快速测试：

| 文件 | 说明 |
|------|------|
| `features_sample.csv` | 特征数据样本，可直接用于模型预测 |
| `prediction_sample.csv` | 预测结果样本格式 |
| `models/om_fault_predictor.pkl` | 预训练模型，可直接加载使用 |