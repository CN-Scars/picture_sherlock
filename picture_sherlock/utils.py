"""
工具函数模块：提供辅助功能、常量定义和通用函数
"""
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
from PIL import Image
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

def validate_paths(paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    验证提供的路径列表，区分有效路径和无效路径
    
    Args:
        paths: 需要验证的路径列表
        
    Returns:
        Tuple[List[str], List[str]]: 有效路径列表和无效路径列表的元组
    """
    valid_paths = []
    invalid_paths = []
    
    for path in paths:
        path = path.strip()
        if path and os.path.isdir(path):
            valid_paths.append(path)
        else:
            invalid_paths.append(path)
            
    return valid_paths, invalid_paths

def get_cache_path_for_folders(folder_paths: List[str]) -> str:
    """
    根据文件夹路径生成一个缓存文件名
    
    Args:
        folder_paths: 文件夹路径列表
        
    Returns:
        str: 缓存文件路径
    """
    # 使用文件夹路径的哈希值作为缓存标识符
    paths_str = ';'.join(sorted(folder_paths))
    hash_id = str(hash(paths_str))[-8:]  # 取哈希值的最后8位作为标识
    
    return os.path.join(CACHE_DIR, f"image_index_{hash_id}")

def get_model_name_from_type(model_type: str) -> str:
    """
    根据模型类型获取对应的模型名称
    
    Args:
        model_type: 模型类型 ('clip' 或 'resnet')
        
    Returns:
        str: 模型名称
    """
    model_mapping = {
        'clip': 'openai/clip-vit-base-patch32',
        'resnet': 'resnet50'
    }
    return model_mapping.get(model_type.lower(), model_mapping['clip']) 