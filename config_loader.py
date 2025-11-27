import json
import os

def load_config(config_path="config.json"):
    """加载JSON配置文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, config_path)
    
    with open(full_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

# 全局配置对象
CONFIG = load_config()