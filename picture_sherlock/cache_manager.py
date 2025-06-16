"""
缓存管理模块：负责持久化缓存的管理、查找和操作
"""
import os
import json
import uuid
import time
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union

# 优先使用绝对导入
try:
    # 尝试绝对导入
    from picture_sherlock.utils import logger, CACHE_DIR
except ImportError:
    # 尝试直接导入
    from utils import logger, CACHE_DIR


class CacheManager:
    """
    缓存管理器类：负责持久化缓存的管理、查找和操作
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化缓存管理器
        
        Args:
            base_dir: 缓存的基础目录，默认为 data/cache/
        """
        self.base_dir = base_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')
        # 确保缓存目录存在
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 确保索引文件存在
        self.index_path = os.path.join(self.base_dir, 'index.json')
        if not os.path.exists(self.index_path):
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
    def get_all_caches(self) -> List[Dict[str, Any]]:
        """
        获取所有缓存条目
        
        Returns:
            List[Dict]: 所有缓存条目的列表
        """
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # 如果索引文件不存在或无效，返回空列表
            return []
    
    def find_cache(self, model: str, dir_paths: Union[str, List[str]]) -> Optional[Dict[str, Any]]:
        """
        根据模型和目录路径查找匹配的缓存条目
        
        Args:
            model: 模型名称
            dir_paths: 目录路径或目录路径列表
        
        Returns:
            Optional[Dict]: 匹配的缓存条目，如果没有找到则返回None
        """
        # 确保dir_paths是列表
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]
            
        # 对目录路径进行排序，确保顺序一致性
        dir_paths = sorted(os.path.abspath(d) for d in dir_paths if os.path.exists(d))
        if not dir_paths:
            return None
            
        # 获取所有缓存条目
        caches = self.get_all_caches()
        
        # 查找匹配的缓存条目
        for cache in caches:
            if cache['model'] == model:
                # 确保cache['dir']也是列表并且排序
                cache_dirs = sorted(cache['dir']) if isinstance(cache['dir'], list) else [cache['dir']]
                if cache_dirs == dir_paths:
                    return cache
                    
        return None
    
    def create_cache(self, model: str, dir_paths: Union[str, List[str]]) -> Dict[str, Any]:
        """
        创建一个新的缓存条目
        
        Args:
            model: 模型名称
            dir_paths: 目录路径或目录路径列表
        
        Returns:
            Dict: 新创建的缓存条目
        """
        # 确保dir_paths是列表
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]
            
        # 对目录路径进行排序，确保顺序一致性
        dir_paths = sorted(os.path.abspath(d) for d in dir_paths if os.path.exists(d))
        
        # 检查是否已存在匹配的缓存
        existing_cache = self.find_cache(model, dir_paths)
        if existing_cache:
            return existing_cache
        
        # 生成唯一的缓存ID
        cache_id = str(uuid.uuid4())
        
        # 创建缓存条目
        cache_entry = {
            'id': cache_id,
            'model': model,
            'dir': dir_paths,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'size': 0,  # 初始大小为0
            'status': 'created'  # 初始状态
        }
        
        # 创建缓存目录
        cache_dir = self.get_cache_directory(cache_id)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 更新索引
        caches = self.get_all_caches()
        caches.append(cache_entry)
        self._save_index(caches)
        
        return cache_entry
    
    def update_cache_status(self, cache_id: str, status: str) -> bool:
        """
        更新缓存状态
        
        Args:
            cache_id: 缓存ID
            status: 新状态
        
        Returns:
            bool: 更新是否成功
        """
        caches = self.get_all_caches()
        for cache in caches:
            if cache['id'] == cache_id:
                cache['status'] = status
                cache['updated_at'] = datetime.now().isoformat()
                self._save_index(caches)
                return True
        return False
    
    def update_cache_size(self, cache_id: str, size: int) -> bool:
        """
        更新缓存大小
        
        Args:
            cache_id: 缓存ID
            size: 新大小（字节）
        
        Returns:
            bool: 更新是否成功
        """
        caches = self.get_all_caches()
        for cache in caches:
            if cache['id'] == cache_id:
                cache['size'] = size
                cache['updated_at'] = datetime.now().isoformat()
                self._save_index(caches)
                return True
        return False
    
    def delete_cache(self, cache_id: str) -> bool:
        """
        删除指定ID的缓存
        
        Args:
            cache_id: 要删除的缓存ID
        
        Returns:
            bool: 删除是否成功
        """
        # 获取所有缓存
        caches = self.get_all_caches()
        
        # 查找并移除匹配的缓存
        found = False
        for i, cache in enumerate(caches):
            if cache['id'] == cache_id:
                del caches[i]
                found = True
                break
        
        if found:
            # 更新索引
            self._save_index(caches)
            
            # 删除缓存目录及其内容
            cache_dir = self.get_cache_directory(cache_id)
            try:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                
                # 删除可能的独立缓存文件
                base_path = self.get_cache_base_path(cache_id)
                for ext in ['_paths.pkl', '_features.npy']:
                    file_path = f"{base_path}{ext}"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                return True
            except Exception as e:
                print(f"删除缓存文件时出错: {e}")
                return False
        
        return False
    
    def get_cache_size(self, cache_id: str) -> int:
        """
        计算缓存的大小（字节）
        
        Args:
            cache_id: 缓存ID
        
        Returns:
            int: 缓存大小（字节）
        """
        total_size = 0
        
        # 获取缓存目录
        cache_dir = self.get_cache_directory(cache_id)
        
        # 计算缓存目录大小
        if os.path.exists(cache_dir):
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        
        # 计算可能的独立缓存文件大小
        base_path = self.get_cache_base_path(cache_id)
        for ext in ['_paths.pkl', '_features.npy']:
            file_path = f"{base_path}{ext}"
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        
        return total_size
    
    def get_cache_directory(self, cache_id: str) -> str:
        """
        获取缓存目录路径
        
        Args:
            cache_id: 缓存ID
        
        Returns:
            str: 缓存目录路径
        """
        return os.path.join(self.base_dir, cache_id)
    
    def get_cache_base_path(self, cache_id: str) -> str:
        """
        获取缓存基础文件路径（不包含扩展名）
        
        Args:
            cache_id: 缓存ID
        
        Returns:
            str: 缓存基础文件路径
        """
        return os.path.join(self.base_dir, cache_id, cache_id)
    
    def clear_all_caches(self, model_type: Optional[str] = None) -> int:
        """
        清除所有缓存或指定模型类型的缓存
        
        Args:
            model_type: 可选，指定要清除的模型类型
        
        Returns:
            int: 已清除的缓存数量
        """
        caches = self.get_all_caches()
        deleted_count = 0
        
        # 收集要删除的缓存ID
        to_delete = []
        for cache in caches:
            if model_type is None or cache['model'] == model_type:
                to_delete.append(cache['id'])
        
        # 删除缓存
        for cache_id in to_delete:
            if self.delete_cache(cache_id):
                deleted_count += 1
        
        return deleted_count
    
    def rebuild_cache(self, cache_id: str) -> bool:
        """
        重建缓存
        
        Args:
            cache_id: 缓存ID
        
        Returns:
            bool: 重建是否成功
        """
        # 获取缓存信息
        caches = self.get_all_caches()
        target_cache = None
        for cache in caches:
            if cache['id'] == cache_id:
                target_cache = cache
                break
        
        if not target_cache:
            return False
        
        # 更新缓存状态为重建中
        self.update_cache_status(cache_id, 'rebuilding')
        
        try:
            # 删除现有缓存文件
            cache_dir = self.get_cache_directory(cache_id)
            if os.path.exists(cache_dir):
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
            
            # 更新缓存时间
            for cache in caches:
                if cache['id'] == cache_id:
                    cache['updated_at'] = datetime.now().isoformat()
                    cache['status'] = 'rebuilt'
                    break
            
            self._save_index(caches)
            return True
        except Exception as e:
            print(f"重建缓存时出错: {e}")
            return False
        
    def batch_delete_caches(self, cache_ids: List[str]) -> Tuple[int, int]:
        """
        批量删除缓存
        
        Args:
            cache_ids: 要删除的缓存ID列表
        
        Returns:
            Tuple[int, int]: (成功删除数量, 失败删除数量)
        """
        success_count = 0
        fail_count = 0
        
        for cache_id in cache_ids:
            if self.delete_cache(cache_id):
                success_count += 1
            else:
                fail_count += 1
        
        return (success_count, fail_count)
    
    def batch_rebuild_caches(self, cache_ids: List[str]) -> Tuple[int, int]:
        """
        批量重建缓存
        
        Args:
            cache_ids: 要重建的缓存ID列表
        
        Returns:
            Tuple[int, int]: (成功重建数量, 失败重建数量)
        """
        success_count = 0
        fail_count = 0
        
        for cache_id in cache_ids:
            if self.rebuild_cache(cache_id):
                success_count += 1
            else:
                fail_count += 1
        
        return (success_count, fail_count)
    
    def search_caches(self, keyword: str) -> List[Dict[str, Any]]:
        """
        根据关键词搜索缓存
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            List[Dict]: 匹配的缓存条目列表
        """
        if not keyword:
            return self.get_all_caches()
        
        keyword = keyword.lower()
        caches = self.get_all_caches()
        results = []
        
        for cache in caches:
            # 在模型名称中搜索
            if keyword in cache['model'].lower():
                results.append(cache)
                continue
                
            # 在目录路径中搜索
            if isinstance(cache['dir'], list):
                for dir_path in cache['dir']:
                    if keyword in str(dir_path).lower():
                        results.append(cache)
                        break
            elif keyword in str(cache['dir']).lower():
                results.append(cache)
        
        return results
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息统计
        
        Returns:
            Dict: 缓存统计信息
        """
        caches = self.get_all_caches()
        
        # 初始化统计变量
        total_size = 0
        model_counts = {}
        directory_stats = {}
        status_counts = {'created': 0, 'indexed': 0, 'failed': 0, 'rebuilding': 0, 'rebuilt': 0}
        
        # 收集统计信息
        for cache in caches:
            # 更新缓存实际大小
            cache_size = self.get_cache_size(cache['id'])
            
            # 更新总大小
            total_size += cache_size
            
            # 更新模型计数
            model_name = cache['model']
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
            
            # 更新状态计数
            status = cache.get('status', 'created')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # 更新目录统计
            if isinstance(cache['dir'], list):
                for dir_path in cache['dir']:
                    directory_stats[dir_path] = directory_stats.get(dir_path, 0) + 1
            else:
                directory_stats[cache['dir']] = directory_stats.get(cache['dir'], 0) + 1
        
        # 构建统计结果
        info = {
            'total_caches': len(caches),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'model_counts': model_counts,
            'directory_stats': directory_stats,
            'status_counts': status_counts,
            'last_updated': datetime.now().isoformat()
        }
        
        return info
    
    def _save_index(self, caches: List[Dict[str, Any]]) -> None:
        """
        保存缓存索引到文件
        
        Args:
            caches: 缓存条目列表
        """
        # 确保缓存目录存在
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 将索引保存到临时文件，然后重命名以确保原子性写入
        temp_path = f"{self.index_path}.temp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(caches, f, indent=2)
        
        # 在Windows上，可能需要首先删除目标文件
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
            
        # 重命名临时文件
        os.rename(temp_path, self.index_path) 