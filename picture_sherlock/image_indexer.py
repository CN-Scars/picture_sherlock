"""
图像索引模块：负责扫描图像文件、构建特征索引、管理缓存
"""
import os
import sys
import time
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# 添加导入模块的路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 优先使用绝对导入
try:
    # 尝试绝对导入
    from picture_sherlock.utils import logger, SUPPORTED_EXTENSIONS, get_cache_path_for_folders
    from picture_sherlock.feature_extractor import FeatureExtractor
    from picture_sherlock.cache_manager import CacheManager
except ImportError:
    # 尝试直接导入
    from utils import logger, SUPPORTED_EXTENSIONS, get_cache_path_for_folders
    from feature_extractor import FeatureExtractor
    from cache_manager import CacheManager


class ImageIndexer:
    """
    图像索引器类：扫描文件夹、提取特征并构建索引
    """
    
    def __init__(self, model_type: str = 'clip'):
        """
        初始化图像索引器
        
        Args:
            model_type: 使用的模型类型 ('clip' 或 'resnet')
        """
        self.model_type = model_type
        self.feature_extractor = None
        self.image_paths = []
        self.features = None
        self.index_built = False
        self.cache_manager = CacheManager()
        self.current_cache_id = None
        
    def scan_image_files(self, folder_paths: List[str]) -> List[str]:
        """
        扫描指定文件夹中的所有图像文件
        
        Args:
            folder_paths: 要扫描的文件夹路径列表
            
        Returns:
            List[str]: 所有找到的图像文件路径列表
        """
        logger.info(f"正在扫描指定文件夹中的图像文件...")
        image_paths = []
        
        for folder_path in folder_paths:
            try:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.splitext(file_path.lower())[1] in SUPPORTED_EXTENSIONS:
                            image_paths.append(file_path)
            except Exception as e:
                logger.error(f"扫描文件夹 {folder_path} 时出错: {e}")
                
        logger.info(f"找到 {len(image_paths)} 张图像文件")
        return image_paths
        
    def _load_cached_index(self, cache_entry: Dict[str, Any]) -> bool:
        """
        尝试加载缓存的索引
        
        Args:
            cache_entry: 缓存条目信息
            
        Returns:
            bool: 如果成功加载缓存则返回True，否则返回False
        """
        try:
            cache_id = cache_entry['id']
            self.current_cache_id = cache_id
            
            # 获取缓存文件路径
            cache_path = self.cache_manager.get_cache_base_path(cache_id)
            
            # 加载图像路径
            paths_file = f"{cache_path}_paths.pkl"
            features_file = f"{cache_path}_features.npy"
            
            if os.path.exists(paths_file) and os.path.exists(features_file):
                with open(paths_file, 'rb') as f:
                    self.image_paths = pickle.load(f)
                    
                self.features = np.load(features_file)
                
                if len(self.image_paths) > 0 and self.features is not None:
                    logger.info(f"从缓存加载了索引，包含 {len(self.image_paths)} 张图像，使用的缓存ID: {cache_id}")
                    self.index_built = True
                    return True
                    
        except Exception as e:
            logger.error(f"加载缓存索引失败: {e}")
            
        return False
        
    def _save_index_to_cache(self, cache_entry: Dict[str, Any]) -> bool:
        """
        将当前索引保存到缓存
        
        Args:
            cache_entry: 缓存条目信息
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        try:
            cache_id = cache_entry['id']
            self.current_cache_id = cache_id
            
            # 获取缓存文件路径
            cache_path = self.cache_manager.get_cache_base_path(cache_id)
            
            # 保存图像路径
            paths_file = f"{cache_path}_paths.pkl"
            features_file = f"{cache_path}_features.npy"
            
            with open(paths_file, 'wb') as f:
                pickle.dump(self.image_paths, f)
                
            # 保存特征向量
            np.save(features_file, self.features)
            
            logger.info(f"已将索引缓存到: {cache_path}，缓存ID: {cache_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存索引到缓存失败: {e}")
            return False
            
    def build_feature_index(self, 
                           folder_paths: List[str], 
                           force_rebuild: bool = False,
                           max_workers: int = 4,
                           progress_callback=None) -> bool:
        """
        构建图像特征索引
        
        Args:
            folder_paths: 要扫描的文件夹路径列表
            force_rebuild: 是否强制重建索引而不使用缓存
            max_workers: 用于并行处理的最大线程数
            progress_callback: 进度回调函数，接收一个0-1之间的浮点数表示进度
            
        Returns:
            bool: 索引构建成功返回True，否则返回False
        """
        start_time = time.time()
        
        try:
            # 如果没有设置特征提取器，则初始化一个
            if self.feature_extractor is None:
                self.feature_extractor = FeatureExtractor(self.model_type)
                
            # 获取模型名称
            model_name = self.feature_extractor.get_model_name()
            
            # 尝试从缓存加载，除非强制重建
            if not force_rebuild:
                # 查找匹配的缓存
                cache_entry = self.cache_manager.find_cache(model_name, folder_paths)
                if cache_entry and self._load_cached_index(cache_entry):
                    logger.info(f"成功从缓存加载索引，使用的模型: {model_name}")
                    return True
                    
            # 扫描图像文件
            self.image_paths = self.scan_image_files(folder_paths)
            
            if not self.image_paths:
                logger.warning("未找到任何图像文件")
                return False
                
            # 提取特征
            logger.info(f"开始为{len(self.image_paths)}张图像提取特征...")
            
            # 创建空的特征数组
            features_list = []
            valid_paths = []
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.feature_extractor, path): path for path in self.image_paths}
                
                total = len(futures)
                completed = 0
                
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        features = future.result()
                        if features is not None:
                            features_list.append(features)
                            valid_paths.append(path)
                    except Exception as e:
                        logger.error(f"处理图像 {path} 时出错: {e}")
                        
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / total)
                        
            # 过滤掉处理失败的图像
            self.image_paths = valid_paths
            
            if not features_list:
                logger.warning("没有提取出任何有效的特征")
                return False
                
            # 将特征列表转换为numpy数组
            self.features = np.vstack(features_list)
            self.index_built = True
            
            # 创建新的缓存条目并保存
            cache_entry = self.cache_manager.create_cache(model_name, folder_paths)
            self._save_index_to_cache(cache_entry)
            
            elapsed_time = time.time() - start_time
            logger.info(f"特征索引构建完成，处理了 {len(self.image_paths)} 张图像，耗时 {elapsed_time:.2f} 秒")
            return True
            
        except Exception as e:
            logger.error(f"构建特征索引失败: {e}")
            return False
            
    def get_index(self) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        获取当前索引
        
        Returns:
            Tuple[List[str], Optional[np.ndarray]]: 图像路径列表和特征数组的元组
        """
        if not self.index_built:
            logger.warning("索引尚未构建")
            return [], None
            
        return self.image_paths, self.features 