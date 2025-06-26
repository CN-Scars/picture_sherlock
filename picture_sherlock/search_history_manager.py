"""
搜索历史记录管理器
负责搜索历史的存储、查询、管理和统计功能
"""
import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import shutil
from PIL import Image

# 优先使用绝对导入
try:
    from picture_sherlock.utils import logger, CACHE_DIR
except ImportError:
    from utils import logger, CACHE_DIR

# 搜索历史配置
SEARCH_HISTORY_CONFIG = {
    "max_records": 5000,  # 最大记录数
    "auto_cleanup_days": 365,  # 自动清理天数
    "backup_enabled": True,  # 启用备份
    "enable_statistics": True,  # 启用统计功能
    "backup_interval_days": 7,  # 备份间隔天数
}

# 查询图像存储配置
QUERY_IMAGE_CONFIG = {
    "max_image_size": (800, 800),  # 最大图像尺寸
    "image_quality": 85,  # JPEG质量
    "max_stored_images": 1000,  # 最大存储图像数
    "storage_subdir": "query_images",  # 存储子目录名
}


class SearchHistoryManager:
    """
    搜索历史记录管理器类
    提供搜索历史的完整生命周期管理功能
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化搜索历史管理器
        
        Args:
            storage_dir: 存储目录，默认使用 CACHE_DIR
        """
        self.storage_dir = storage_dir or CACHE_DIR
        self.history_file = os.path.join(self.storage_dir, 'search_history.json')
        self.backup_dir = os.path.join(self.storage_dir, 'backups')
        self.query_images_dir = os.path.join(self.storage_dir, QUERY_IMAGE_CONFIG['storage_subdir'])

        # 确保目录存在
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.query_images_dir, exist_ok=True)
        
        # 初始化历史文件
        self._initialize_history_file()
        
        logger.info(f"SearchHistoryManager initialized with storage: {self.storage_dir}")
    
    def _initialize_history_file(self):
        """初始化历史记录文件"""
        if not os.path.exists(self.history_file):
            initial_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "config": SEARCH_HISTORY_CONFIG,
                "search_history": []
            }
            self._save_history_data(initial_data)
            logger.info("Created new search history file")
    
    def _load_history_data(self) -> Dict[str, Any]:
        """
        加载历史记录数据
        
        Returns:
            Dict: 历史记录数据
        """
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 数据格式验证
                if not isinstance(data, dict) or 'search_history' not in data:
                    raise ValueError("Invalid history file format")
                return data
        except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to load history data: {e}")
            # 尝试从备份恢复
            if self._restore_from_backup():
                return self._load_history_data()
            else:
                # 创建新的历史文件
                self._initialize_history_file()
                return self._load_history_data()
    
    def _save_history_data(self, data: Dict[str, Any]):
        """
        保存历史记录数据
        
        Args:
            data: 要保存的数据
        """
        try:
            data['last_updated'] = datetime.now().isoformat()
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history data: {e}")
            raise
    
    def _generate_image_hash(self, image_path: str) -> Optional[str]:
        """
        生成图像文件的MD5哈希值
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: MD5哈希值，如果文件不存在返回None
        """
        try:
            if not os.path.exists(image_path):
                return None
            
            hash_md5 = hashlib.md5()
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate hash for {image_path}: {e}")
            return None
    
    def add_search_record(self,
                         query_image_path: str,
                         target_folders: List[str],
                         model_type: str,
                         model_name: str,
                         similarity_threshold: float,
                         results_count: int,
                         execution_time: float,
                         max_results: int = 50,
                         top_similarity: Optional[float] = None,
                         search_results: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        添加新的搜索记录

        Args:
            query_image_path: 查询图像路径
            target_folders: 目标文件夹列表
            model_type: 模型类型
            model_name: 模型名称
            similarity_threshold: 相似度阈值
            results_count: 结果数量
            execution_time: 执行时间
            max_results: 最大结果数
            top_similarity: 最高相似度
            search_results: 搜索结果列表，包含路径和相似度

        Returns:
            str: 新记录的ID
        """
        try:
            # 生成记录ID
            record_id = str(uuid.uuid4())
            
            # 保存查询图像并获取图像信息
            stored_image_path = self.save_query_image(query_image_path, record_id)

            image_info = {
                "size": os.path.getsize(query_image_path) if os.path.exists(query_image_path) else 0,
                "hash": self._generate_image_hash(query_image_path),
                "upload_time": datetime.now().isoformat(),
                "stored_path": stored_image_path  # 保存的图像相对路径
            }
            
            # 创建搜索记录
            search_record = {
                "id": record_id,
                "timestamp": datetime.now().isoformat(),
                "query_image": image_info,
                "search_config": {
                    "target_folders": target_folders,
                    "model_type": model_type,
                    "model_name": model_name,
                    "similarity_threshold": similarity_threshold,
                    "max_results": max_results
                },
                "results": {
                    "count": results_count,
                    "execution_time": execution_time,
                    "top_similarity": top_similarity,
                    "similar_images": search_results or []
                },
                "user_data": {
                    "tags": [],
                    "notes": "",
                    "favorite": False,
                    "rating": None
                }
            }
            
            # 加载现有数据
            data = self._load_history_data()
            data['search_history'].append(search_record)
            
            # 检查是否需要清理旧记录
            self._cleanup_old_records(data)
            
            # 保存数据
            self._save_history_data(data)
            
            # 创建备份
            if SEARCH_HISTORY_CONFIG['backup_enabled']:
                self._create_backup_if_needed()
            
            logger.info(f"Added search record: {record_id}")
            return record_id

        except Exception as e:
            logger.error(f"Failed to add search record: {e}")
            raise

    def add_search_record_with_existing_image(self,
                                            existing_image_path: str,
                                            target_folders: List[str],
                                            model_type: str,
                                            model_name: str,
                                            similarity_threshold: float,
                                            results_count: int,
                                            execution_time: float,
                                            max_results: int = 50,
                                            top_similarity: Optional[float] = None,
                                            search_results: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        添加新的搜索记录，但复用现有的查询图像

        Args:
            existing_image_path: 现有图像的相对路径（用于复用）
            target_folders: 目标文件夹列表
            model_type: 模型类型
            model_name: 模型名称
            similarity_threshold: 相似度阈值
            results_count: 结果数量
            execution_time: 执行时间
            max_results: 最大结果数
            top_similarity: 最高相似度
            search_results: 搜索结果列表，包含路径和相似度

        Returns:
            str: 新记录的ID
        """
        try:
            # 生成记录ID
            record_id = str(uuid.uuid4())

            # 获取现有图像的完整路径用于计算hash和大小
            if not os.path.isabs(existing_image_path):
                full_image_path = os.path.join(self.storage_dir, existing_image_path)
            else:
                full_image_path = existing_image_path

            # 为新记录创建图像链接（硬链接或复制）
            new_stored_path = self._create_image_link(full_image_path, record_id)

            # 复用现有图像信息但使用新的路径
            image_info = {
                "size": os.path.getsize(full_image_path) if os.path.exists(full_image_path) else 0,
                "hash": self._generate_image_hash(full_image_path),
                "upload_time": datetime.now().isoformat(),
                "stored_path": new_stored_path  # 使用新记录的相对路径
            }

            # 创建搜索记录
            search_record = {
                "id": record_id,
                "timestamp": datetime.now().isoformat(),
                "query_image": image_info,
                "search_config": {
                    "target_folders": target_folders,
                    "model_type": model_type,
                    "model_name": model_name,
                    "similarity_threshold": similarity_threshold,
                    "max_results": max_results
                },
                "results": {
                    "count": results_count,
                    "execution_time": execution_time,
                    "top_similarity": top_similarity,
                    "similar_images": search_results or []
                },
                "user_data": {
                    "tags": [],
                    "notes": "",
                    "favorite": False,
                    "rating": None
                }
            }

            # 加载现有数据
            data = self._load_history_data()
            data['search_history'].append(search_record)

            # 检查是否需要清理旧记录
            self._cleanup_old_records(data)

            # 保存数据
            self._save_history_data(data)

            # 创建备份
            if SEARCH_HISTORY_CONFIG['backup_enabled']:
                self._create_backup_if_needed()

            logger.info(f"Added search record with existing image: {record_id}")
            return record_id

        except Exception as e:
            logger.error(f"Failed to add search record with existing image: {e}")
            raise
    
    def get_all_records(self, limit: Optional[int] = None, 
                       offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取所有搜索记录
        
        Args:
            limit: 限制返回数量
            offset: 偏移量
            
        Returns:
            List[Dict]: 搜索记录列表
        """
        try:
            data = self._load_history_data()
            records = data['search_history']
            
            # 按时间倒序排列
            records.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # 应用分页
            if limit is not None:
                records = records[offset:offset + limit]
            
            return records
        except Exception as e:
            logger.error(f"Failed to get records: {e}")
            return []
    
    def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取搜索记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            Dict: 搜索记录，如果不存在返回None
        """
        try:
            data = self._load_history_data()
            for record in data['search_history']:
                if record['id'] == record_id:
                    return record
            return None
        except Exception as e:
            logger.error(f"Failed to get record {record_id}: {e}")
            return None
    
    def _cleanup_old_records(self, data: Dict[str, Any]):
        """
        清理旧的搜索记录
        
        Args:
            data: 历史数据
        """
        max_records = SEARCH_HISTORY_CONFIG['max_records']
        cleanup_days = SEARCH_HISTORY_CONFIG['auto_cleanup_days']
        
        records = data['search_history']
        
        # 按数量限制清理
        if len(records) > max_records:
            # 保留最新的记录
            records.sort(key=lambda x: x['timestamp'], reverse=True)
            data['search_history'] = records[:max_records]
            logger.info(f"Cleaned up {len(records) - max_records} records by count limit")
        
        # 按时间清理
        if cleanup_days > 0:
            cutoff_date = datetime.now() - timedelta(days=cleanup_days)
            original_count = len(data['search_history'])
            data['search_history'] = [
                record for record in data['search_history']
                if datetime.fromisoformat(record['timestamp']) > cutoff_date
            ]
            cleaned_count = original_count - len(data['search_history'])
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} records by time limit")
    
    def _create_backup_if_needed(self):
        """根据需要创建备份"""
        try:
            backup_file = os.path.join(
                self.backup_dir, 
                f"search_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # 检查是否需要创建备份
            if self._should_create_backup():
                shutil.copy2(self.history_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
                
                # 清理旧备份
                self._cleanup_old_backups()
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def _should_create_backup(self) -> bool:
        """检查是否应该创建备份"""
        backup_interval = SEARCH_HISTORY_CONFIG['backup_interval_days']
        
        # 获取最新备份时间
        backup_files = [f for f in os.listdir(self.backup_dir) 
                       if f.startswith('search_history_backup_')]
        
        if not backup_files:
            return True
        
        # 获取最新备份文件的时间
        latest_backup = max(backup_files)
        backup_time_str = latest_backup.replace('search_history_backup_', '').replace('.json', '')
        
        try:
            backup_time = datetime.strptime(backup_time_str, '%Y%m%d_%H%M%S')
            return (datetime.now() - backup_time).days >= backup_interval
        except ValueError:
            return True
    
    def _cleanup_old_backups(self):
        """清理旧的备份文件"""
        try:
            backup_files = [f for f in os.listdir(self.backup_dir) 
                           if f.startswith('search_history_backup_')]
            
            # 保留最近的10个备份
            if len(backup_files) > 10:
                backup_files.sort(reverse=True)
                for old_backup in backup_files[10:]:
                    os.remove(os.path.join(self.backup_dir, old_backup))
                logger.info(f"Cleaned up {len(backup_files) - 10} old backups")
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")
    
    def _restore_from_backup(self) -> bool:
        """
        从备份恢复历史记录
        
        Returns:
            bool: 恢复是否成功
        """
        try:
            backup_files = [f for f in os.listdir(self.backup_dir) 
                           if f.startswith('search_history_backup_')]
            
            if not backup_files:
                return False
            
            # 使用最新的备份
            latest_backup = max(backup_files)
            backup_path = os.path.join(self.backup_dir, latest_backup)
            
            shutil.copy2(backup_path, self.history_file)
            logger.info(f"Restored from backup: {latest_backup}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

    def get_records_by_filter(self,
                             model_type: Optional[str] = None,
                             date_from: Optional[datetime] = None,
                             date_to: Optional[datetime] = None,
                             target_folder: Optional[str] = None,
                             tags: Optional[List[str]] = None,
                             favorites_only: bool = False,
                             limit: Optional[int] = None,
                             offset: int = 0) -> List[Dict[str, Any]]:
        """
        根据条件筛选搜索记录

        Args:
            model_type: 模型类型筛选
            date_from: 开始日期
            date_to: 结束日期
            target_folder: 目标文件夹筛选
            tags: 标签筛选
            favorites_only: 仅显示收藏
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            List[Dict]: 筛选后的搜索记录列表
        """
        try:
            data = self._load_history_data()
            records = data['search_history']

            # 应用筛选条件
            filtered_records = []
            for record in records:
                # 模型类型筛选
                if model_type and record['search_config']['model_type'] != model_type:
                    continue

                # 日期筛选
                record_date = datetime.fromisoformat(record['timestamp'])
                if date_from and record_date < date_from:
                    continue
                if date_to and record_date > date_to:
                    continue

                # 目标文件夹筛选
                if target_folder:
                    if target_folder not in record['search_config']['target_folders']:
                        continue

                # 标签筛选
                if tags:
                    record_tags = record['user_data']['tags']
                    if not any(tag in record_tags for tag in tags):
                        continue

                # 收藏筛选
                if favorites_only and not record['user_data']['favorite']:
                    continue

                filtered_records.append(record)

            # 按时间倒序排列
            filtered_records.sort(key=lambda x: x['timestamp'], reverse=True)

            # 应用分页
            if limit is not None:
                filtered_records = filtered_records[offset:offset + limit]

            return filtered_records
        except Exception as e:
            logger.error(f"Failed to filter records: {e}")
            return []

    def update_record(self, record_id: str, **kwargs) -> bool:
        """
        更新搜索记录

        Args:
            record_id: 记录ID
            **kwargs: 要更新的字段

        Returns:
            bool: 更新是否成功
        """
        try:
            data = self._load_history_data()

            for record in data['search_history']:
                if record['id'] == record_id:
                    # 更新用户数据字段
                    if 'tags' in kwargs:
                        record['user_data']['tags'] = kwargs['tags']
                    if 'notes' in kwargs:
                        record['user_data']['notes'] = kwargs['notes']
                    if 'favorite' in kwargs:
                        record['user_data']['favorite'] = kwargs['favorite']
                    if 'rating' in kwargs:
                        record['user_data']['rating'] = kwargs['rating']

                    # 保存数据
                    self._save_history_data(data)
                    logger.info(f"Updated record: {record_id}")
                    return True

            logger.warning(f"Record not found: {record_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to update record {record_id}: {e}")
            return False

    def delete_record(self, record_id: str) -> bool:
        """
        删除搜索记录

        Args:
            record_id: 记录ID

        Returns:
            bool: 删除是否成功
        """
        try:
            data = self._load_history_data()
            original_count = len(data['search_history'])

            data['search_history'] = [
                record for record in data['search_history']
                if record['id'] != record_id
            ]

            if len(data['search_history']) < original_count:
                # 同时删除对应的查询图像
                self.delete_query_image(record_id)
                self._save_history_data(data)
                logger.info(f"Deleted record and query image: {record_id}")
                return True
            else:
                logger.warning(f"Record not found: {record_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {e}")
            return False

    def clear_all_records(self) -> bool:
        """
        清空所有搜索记录

        Returns:
            bool: 清空是否成功
        """
        try:
            data = self._load_history_data()
            record_count = len(data['search_history'])

            # 删除所有查询图像
            for record in data['search_history']:
                self.delete_query_image(record['id'])

            data['search_history'] = []
            self._save_history_data(data)
            logger.info(f"Cleared {record_count} records and their query images")
            return True
        except Exception as e:
            logger.error(f"Failed to clear records: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取搜索历史统计信息

        Returns:
            Dict: 统计信息
        """
        try:
            data = self._load_history_data()
            records = data['search_history']

            if not records:
                return {
                    "total_searches": 0,
                    "model_usage": {},
                    "average_execution_time": 0,
                    "favorite_count": 0,
                    "date_range": None,
                    "most_searched_folders": {},
                    "search_frequency": {}
                }

            # 基础统计
            total_searches = len(records)
            favorite_count = sum(1 for r in records if r['user_data']['favorite'])

            # 模型使用统计
            model_usage = {}
            execution_times = []
            folder_usage = {}

            # 按日期统计搜索频率
            search_frequency = {}

            # 标签统计
            tag_usage = {}
            records_with_tags = 0

            for record in records:
                # 模型统计
                model_type = record['search_config']['model_type']
                model_usage[model_type] = model_usage.get(model_type, 0) + 1

                # 执行时间统计
                exec_time = record['results']['execution_time']
                if exec_time:
                    execution_times.append(exec_time)

                # 文件夹使用统计
                for folder in record['search_config']['target_folders']:
                    folder_usage[folder] = folder_usage.get(folder, 0) + 1

                # 搜索频率统计（按日期）
                date_key = record['timestamp'][:10]  # YYYY-MM-DD
                search_frequency[date_key] = search_frequency.get(date_key, 0) + 1

                # 标签统计
                tags = record.get('user_data', {}).get('tags', [])
                if tags:
                    records_with_tags += 1
                    for tag in tags:
                        tag_usage[tag] = tag_usage.get(tag, 0) + 1

            # 计算平均执行时间
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

            # 获取日期范围
            timestamps = [datetime.fromisoformat(r['timestamp']) for r in records]
            date_range = {
                "earliest": min(timestamps).isoformat(),
                "latest": max(timestamps).isoformat()
            } if timestamps else None

            # 最常搜索的文件夹（前5个）
            most_searched_folders = dict(
                sorted(folder_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            )

            # 最常用的标签（前10个）
            most_used_tags = dict(
                sorted(tag_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            return {
                "total_searches": total_searches,
                "model_usage": model_usage,
                "average_execution_time": round(avg_execution_time, 2),
                "favorite_count": favorite_count,
                "date_range": date_range,
                "most_searched_folders": most_searched_folders,
                "search_frequency": search_frequency,
                "tag_statistics": {
                    "total_tags": len(tag_usage),
                    "records_with_tags": records_with_tags,
                    "tag_usage": tag_usage,
                    "most_used_tags": most_used_tags
                }
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def get_all_tags(self) -> Dict[str, int]:
        """
        获取所有标签及其使用次数

        Returns:
            Dict[str, int]: 标签名称和使用次数的字典
        """
        try:
            data = self._load_history_data()
            records = data['search_history']

            tag_usage = {}
            for record in records:
                tags = record.get('user_data', {}).get('tags', [])
                for tag in tags:
                    tag_usage[tag] = tag_usage.get(tag, 0) + 1

            return tag_usage
        except Exception as e:
            logger.error(f"Failed to get all tags: {e}")
            return {}

    def get_records_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        根据标签获取搜索记录

        Args:
            tag: 标签名称

        Returns:
            List[Dict]: 包含该标签的搜索记录列表
        """
        try:
            data = self._load_history_data()
            records = data['search_history']

            tagged_records = []
            for record in records:
                tags = record.get('user_data', {}).get('tags', [])
                if tag in tags:
                    tagged_records.append(record)

            # 按时间倒序排列
            tagged_records.sort(key=lambda x: x['timestamp'], reverse=True)
            return tagged_records
        except Exception as e:
            logger.error(f"Failed to get records by tag {tag}: {e}")
            return []

    def export_history(self, export_path: str,
                      include_user_data: bool = True) -> bool:
        """
        导出搜索历史

        Args:
            export_path: 导出文件路径
            include_user_data: 是否包含用户数据（标签、备注等）

        Returns:
            bool: 导出是否成功
        """
        try:
            data = self._load_history_data()

            if not include_user_data:
                # 移除用户数据
                for record in data['search_history']:
                    record.pop('user_data', None)

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported history to: {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False

    def import_history(self, import_path: str,
                      merge: bool = True) -> bool:
        """
        导入搜索历史

        Args:
            import_path: 导入文件路径
            merge: 是否与现有数据合并

        Returns:
            bool: 导入是否成功
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            if 'search_history' not in import_data:
                logger.error("Invalid import file format")
                return False

            if merge:
                # 合并数据
                current_data = self._load_history_data()

                # 获取现有记录的ID集合
                existing_ids = {r['id'] for r in current_data['search_history']}

                # 添加不重复的记录
                new_records = [
                    record for record in import_data['search_history']
                    if record['id'] not in existing_ids
                ]

                current_data['search_history'].extend(new_records)
                self._save_history_data(current_data)

                logger.info(f"Imported {len(new_records)} new records")
            else:
                # 替换数据
                import_data['last_updated'] = datetime.now().isoformat()
                self._save_history_data(import_data)
                logger.info("Replaced history with imported data")

            return True
        except Exception as e:
            logger.error(f"Failed to import history: {e}")
            return False

    def save_query_image(self, image_path: str, record_id: str) -> Optional[str]:
        """
        保存查询图像到存储目录

        Args:
            image_path: 原始图像路径
            record_id: 记录ID

        Returns:
            str: 保存的图像相对路径，失败返回None
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Query image not found: {image_path}")
                return None

            # 生成存储文件名
            stored_filename = f"{record_id}.jpg"
            stored_path = os.path.join(self.query_images_dir, stored_filename)

            # 打开并处理图像
            with Image.open(image_path) as img:
                # 转换为RGB模式（处理RGBA等格式）
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 调整图像尺寸
                max_size = QUERY_IMAGE_CONFIG['max_image_size']
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # 保存为JPEG格式
                img.save(
                    stored_path,
                    'JPEG',
                    quality=QUERY_IMAGE_CONFIG['image_quality'],
                    optimize=True
                )

            # 返回相对路径
            relative_path = os.path.join(QUERY_IMAGE_CONFIG['storage_subdir'], stored_filename)
            logger.info(f"Saved query image: {relative_path}")
            return relative_path

        except Exception as e:
            logger.error(f"Failed to save query image: {e}")
            return None

    def _create_image_link(self, source_image_path: str, record_id: str) -> Optional[str]:
        """
        为新记录创建图像链接（硬链接或复制）

        Args:
            source_image_path: 源图像的完整路径
            record_id: 新记录的ID

        Returns:
            str: 新图像的相对路径，失败返回None
        """
        try:
            if not os.path.exists(source_image_path):
                logger.warning(f"Source image not found: {source_image_path}")
                return None

            # 生成新的存储文件名
            stored_filename = f"{record_id}.jpg"
            stored_path = os.path.join(self.query_images_dir, stored_filename)

            # 尝试创建硬链接，失败则复制文件
            try:
                os.link(source_image_path, stored_path)
                logger.info(f"Created hard link for query image: {stored_filename}")
            except (OSError, NotImplementedError):
                # 硬链接失败，使用复制
                shutil.copy2(source_image_path, stored_path)
                logger.info(f"Copied query image: {stored_filename}")

            # 返回相对路径
            relative_path = os.path.join(QUERY_IMAGE_CONFIG['storage_subdir'], stored_filename)
            return relative_path

        except Exception as e:
            logger.error(f"Failed to create image link: {e}")
            return None

    def get_query_image_path(self, record_id: str) -> Optional[str]:
        """
        获取查询图像的完整路径

        Args:
            record_id: 记录ID

        Returns:
            str: 图像完整路径，不存在返回None
        """
        try:
            stored_filename = f"{record_id}.jpg"
            stored_path = os.path.join(self.query_images_dir, stored_filename)

            if os.path.exists(stored_path):
                return stored_path
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get query image path: {e}")
            return None

    def delete_query_image(self, record_id: str) -> bool:
        """
        删除查询图像文件

        Args:
            record_id: 记录ID

        Returns:
            bool: 删除成功返回True
        """
        try:
            stored_filename = f"{record_id}.jpg"
            stored_path = os.path.join(self.query_images_dir, stored_filename)

            if os.path.exists(stored_path):
                os.remove(stored_path)
                logger.info(f"Deleted query image: {stored_filename}")
                return True
            else:
                logger.warning(f"Query image not found for deletion: {stored_filename}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete query image: {e}")
            return False

    def cleanup_orphaned_images(self) -> int:
        """
        清理孤儿图像文件（没有对应记录的图像）

        Returns:
            int: 清理的图像数量
        """
        try:
            # 获取所有记录ID
            data = self._load_history_data()
            record_ids = {record['id'] for record in data['search_history']}

            # 获取所有图像文件
            if not os.path.exists(self.query_images_dir):
                return 0

            image_files = [f for f in os.listdir(self.query_images_dir) if f.endswith('.jpg')]

            # 找出孤儿图像
            orphaned_count = 0
            for image_file in image_files:
                # 从文件名提取记录ID
                record_id = os.path.splitext(image_file)[0]

                if record_id not in record_ids:
                    # 这是一个孤儿图像，删除它
                    image_path = os.path.join(self.query_images_dir, image_file)
                    try:
                        os.remove(image_path)
                        orphaned_count += 1
                        logger.info(f"Removed orphaned image: {image_file}")
                    except Exception as e:
                        logger.error(f"Failed to remove orphaned image {image_file}: {e}")

            if orphaned_count > 0:
                logger.info(f"Cleaned up {orphaned_count} orphaned images")

            return orphaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned images: {e}")
            return 0
