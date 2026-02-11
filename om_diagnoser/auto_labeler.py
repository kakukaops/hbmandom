#!/usr/bin/env python3
"""
光模块故障自动标注器

根据用户定义的规则对原始指标特征文件进行自动化标注。
支持自定义故障类型名称与故障标注规则。

Author: liyan
Date: 2025-11-28
"""

import os
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime


class AutoLabeler:
    def __init__(self, config_path: str = "config/rules.yaml"):
        self.config_path = config_path
        self.config = None
        self.rules = []
        self.supported_operators = {
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y,
            ">": lambda x, y: x > y,
            ">=": lambda x, y: x >= y,
            "<": lambda x, y: x < y,
            "<=": lambda x, y: x <= y,
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.load_config()
    
    def load_config(self) -> None:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 验证配置结构
            if not self.config:
                raise ValueError("配置文件为空")
            
            # 提取规则
            if 'rules' in self.config:
                self.rules = self.config['rules']
            
            # 提取支持的运算符（如果有自定义）
            if 'supported_operators' in self.config:
                self.supported_operators.update({
                    op: self._create_operator_func(op)
                    for op in self.config['supported_operators']
                    if op not in self.supported_operators
                })
            
            self.logger.info(f"成功加载配置文件: {self.config_path}")
            self.logger.info(f"加载了 {len(self.rules)} 条标注规则")
            
        except FileNotFoundError:
            self.logger.error(f"配置文件不存在: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"加载配置时发生错误: {e}")
            raise
    
    def _create_operator_func(self, operator: str):
        if operator == "==":
            return lambda x, y: x == y
        elif operator == "!=":
            return lambda x, y: x != y
        elif operator == ">":
            return lambda x, y: x > y
        elif operator == ">=":
            return lambda x, y: x >= y
        elif operator == "<":
            return lambda x, y: x < y
        elif operator == "<=":
            return lambda x, y: x <= y
        else:
            raise ValueError(f"不支持的运算符: {operator}")
    
    def _evaluate_condition(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        column = condition['column']
        operator = condition['operator']
        value = condition['value']
        
        if column not in row:
            self.logger.warning(f"列 '{column}' 不存在于数据中")
            return False
        
        if operator not in self.supported_operators:
            self.logger.warning(f"不支持的运算符: {operator}")
            return False
        
        try:
            return self.supported_operators[operator](row[column], value)
        except Exception as e:
            self.logger.error(f"评估条件时发生错误: {e}")
            return False
    
    def _evaluate_rule(self, row: pd.Series, rule: Dict[str, Any]) -> bool:
        if 'conditions' not in rule:
            self.logger.warning(f"规则 '{rule.get('name', '未知')}' 没有条件")
            return False
        
        conditions = rule['conditions']
        
        # 所有条件都必须满足（AND逻辑）
        for condition in conditions:
            if not self._evaluate_condition(row, condition):
                return False
        
        return True
    
    def label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            self.logger.warning("输入数据为空")
            return data
        
        # 创建标注结果的副本
        labeled_data = data.copy()
        
        # 统计信息
        rule_stats = {rule['name']: 0 for rule in self.rules}
        
        # 应用每条规则
        for rule in self.rules:
            rule_name = rule['name']
            label_column = rule.get('label_column')
            label_value = rule.get('label_value')
            
            if not label_column or label_value is None:
                self.logger.warning(f"规则 '{rule_name}' 缺少标注列或标注值")
                continue
            
            # 确保标注列存在
            if label_column not in labeled_data.columns:
                labeled_data[label_column] = 0
            
            # 应用规则
            for idx, row in labeled_data.iterrows():
                if self._evaluate_rule(row, rule):
                    labeled_data.at[idx, label_column] = label_value
                    rule_stats[rule_name] += 1
        
        # 记录统计信息
        self.logger.info("标注完成，统计信息:")
        for rule_name, count in rule_stats.items():
            self.logger.info(f"  规则 '{rule_name}': {count} 条记录被标注")
        
        total_labeled = sum(rule_stats.values())
        self.logger.info(f"  总计: {total_labeled} 条记录被标注")
        
        return labeled_data
    
    def load_data(self, input_path: Optional[str] = None) -> pd.DataFrame:
        if input_path is None:
            if not self.config or 'files' not in self.config or 'input_path' not in self.config['files']:
                raise ValueError("配置文件中未指定输入文件路径")
            input_path = self.config['files']['input_path']
        
        try:
            data = pd.read_csv(input_path)
            self.logger.info(f"成功加载数据: {input_path}")
            self.logger.info(f"数据形状: {data.shape}")
            self.logger.info(f"数据列: {list(data.columns)}")
            return data
        except FileNotFoundError:
            self.logger.error(f"输入文件不存在: {input_path}")
            raise
        except Exception as e:
            self.logger.error(f"加载数据时发生错误: {e}")
            raise
    
    def save_data(self, data: pd.DataFrame, output_path: Optional[str] = None) -> None:
        if output_path is None:
            if not self.config or 'files' not in self.config or 'output_path' not in self.config['files']:
                raise ValueError("配置文件中未指定输出文件路径")
            output_path = self.config['files']['output_path']
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            data.to_csv(output_path, index=False)
            self.logger.info(f"成功保存标注后的数据: {output_path}")
            self.logger.info(f"保存的数据形状: {data.shape}")
        except Exception as e:
            self.logger.error(f"保存数据时发生错误: {e}")
            raise
    
    def run(self, input_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
        self.logger.info("开始自动标注流程...")
        
        # 加载数据
        data = self.load_data(input_path)
        
        # 标注数据
        labeled_data = self.label_data(data)
        
        # 保存数据
        self.save_data(labeled_data, output_path)
        
        self.logger.info("自动标注流程完成")
        return labeled_data


if __name__ == '__main__':
    print("开始测试自动标注器...")
    
    try:
        # 创建自动标注器实例
        labeler = AutoLabeler("config/rules.yaml")
        
        # 运行自动标注
        labeled_data = labeler.run()
        
        # 显示一些统计信息
        print("\n标注结果统计:")
        print(f"总记录数: {len(labeled_data)}")
        
        # 检查标注列
        label_columns = ['rx_los', 'tx_fault', 'rx_lol', 'fec_burst']
        for col in label_columns:
            if col in labeled_data.columns:
                count = (labeled_data[col] == 1).sum()
                print(f"{col} = 1 的记录数: {count}")
        
        # 显示前几行数据
        print("\n标注后的前5行数据:")
        print(labeled_data.head())
        
        # 保存统计信息
        stats = {
            'total_records': len(labeled_data),
            'label_counts': {col: int((labeled_data[col] == 1).sum()) 
                           for col in label_columns if col in labeled_data.columns},
            'timestamp': datetime.now().isoformat()
        }
        
        stats_path = "data/labeling_stats.json"
        import json
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n统计信息已保存到: {stats_path}")
        
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()