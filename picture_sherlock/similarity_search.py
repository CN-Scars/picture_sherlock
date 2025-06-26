"""
相似度搜索模块：提供图像相似度计算和搜索功能
"""
import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 添加导入模块的路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 优先使用绝对导入
try:
    # 尝试绝对导入
    from picture_sherlock.utils import logger, validate_paths
    from picture_sherlock.feature_extractor import FeatureExtractor
    from picture_sherlock.image_indexer import ImageIndexer
    from picture_sherlock.search_history_manager import SearchHistoryManager
except ImportError:
    # 尝试直接导入
    from utils import logger, validate_paths
    from feature_extractor import FeatureExtractor
    from image_indexer import ImageIndexer
    from search_history_manager import SearchHistoryManager


class SimilaritySearch:
    """
    相似度搜索类：提供图像相似度计算和搜索功能
    """
    
    def __init__(self,
                model_type: str = 'clip',
                model_name: Optional[str] = None,
                download_progress_callback: Optional[Callable[[str, float], None]] = None,
                enable_history: bool = True):
        """
        初始化相似度搜索器

        Args:
            model_type: 使用的模型类型 ('clip', 'resnet' 或 'custom')
            model_name: 自定义模型名称，当model_type为'custom'时必须提供
            download_progress_callback: 模型下载进度回调函数
            enable_history: 是否启用搜索历史记录
        """
        self.model_type = model_type
        self.model_name = model_name
        self.download_progress_callback = download_progress_callback
        self.enable_history = enable_history

        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(
            model_type=model_type,
            model_name=model_name,
            download_progress_callback=download_progress_callback
        )

        # 初始化图像索引器
        self.image_indexer = ImageIndexer(model_type)

        # 初始化搜索历史管理器
        if self.enable_history:
            try:
                self.history_manager = SearchHistoryManager()
            except Exception as e:
                logger.warning(f"Failed to initialize search history manager: {e}")
                self.enable_history = False
                self.history_manager = None
        else:
            self.history_manager = None
        
    def calculate_similarity(self, 
                            query_vector: np.ndarray, 
                            reference_vectors: np.ndarray) -> np.ndarray:
        """
        计算查询向量与参考向量集之间的余弦相似度
        
        Args:
            query_vector: 查询图像的特征向量
            reference_vectors: 参考图像的特征向量集
            
        Returns:
            np.ndarray: 相似度得分数组，形状为 (参考向量数量,)
        """
        # 确保查询向量为2D形状
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, reference_vectors).flatten()
        
        return similarities
        
    def search_similar_images(self,
                             query_image_path: str,
                             folder_paths: List[str],
                             top_n: int = 10,
                             force_rebuild_index: bool = False,
                             progress_callback=None,
                             similarity_threshold: float = 0.0,
                             save_to_history: bool = True,
                             reuse_image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        搜索与查询图像相似的图像

        Args:
            query_image_path: 查询图像的路径
            folder_paths: 要搜索的文件夹路径列表
            top_n: 返回最相似的图像数量
            force_rebuild_index: 是否强制重建索引
            progress_callback: 进度回调函数
            similarity_threshold: 相似度阈值（用于历史记录）
            save_to_history: 是否保存到搜索历史
            reuse_image_path: 复用现有图像的相对路径（用于重新执行搜索）

        Returns:
            List[Dict[str, Any]]: 相似图像结果列表，每个元素包含图像路径和相似度得分
                [
                    {
                        "path": "图像路径",
                        "similarity": 相似度得分,
                    },
                    ...
                ]
        """
        start_time = time.time()
        
        # 验证文件夹路径
        valid_paths, invalid_paths = validate_paths(folder_paths)
        
        if invalid_paths:
            logger.warning(f"以下路径无效或不存在: {invalid_paths}")
            
        if not valid_paths:
            logger.error("没有有效的文件夹路径")
            return []
            
        try:
            # 验证查询图像
            if not os.path.isfile(query_image_path):
                logger.error(f"查询图像不存在: {query_image_path}")
                return []
                
            # 提取查询图像特征
            logger.info(f"提取查询图像特征: {query_image_path}")
            query_features = self.feature_extractor.extract_features(query_image_path)
            
            if query_features is None:
                logger.error("无法从查询图像提取特征")
                return []
                
            # 构建或加载图像索引
            if progress_callback:
                progress_callback(0.1)  # 初始进度
                
            index_build_result = self.image_indexer.build_feature_index(
                valid_paths, 
                force_rebuild=force_rebuild_index,
                progress_callback=lambda p: progress_callback(0.1 + p * 0.8) if progress_callback else None
            )
            
            if not index_build_result:
                logger.error("构建图像索引失败")
                return []
                
            # 获取索引数据
            image_paths, features = self.image_indexer.get_index()
            
            if len(image_paths) == 0 or features is None:
                logger.warning("索引为空或无效")
                return []
                
            # 计算相似度
            logger.info("计算图像相似度...")
            similarities = self.calculate_similarity(query_features, features)
            
            if progress_callback:
                progress_callback(0.95)  # 接近完成
                
            # 获取最相似的图像索引（排除查询图像自身）
            sorted_indices = np.argsort(similarities)[::-1]  # 降序排列
            
            # 构建结果列表
            results = []
            count = 0
            
            for idx in sorted_indices:
                path = image_paths[idx]
                # 排除查询图像自身（如果它在索引中）
                if os.path.normpath(path) == os.path.normpath(query_image_path):
                    continue
                    
                results.append({
                    "path": path,  # 保持绝对路径
                    "similarity": float(similarities[idx])  # 转换为Python标准类型
                })
                
                count += 1
                if count >= top_n:
                    break
                    
            elapsed_time = time.time() - start_time
            logger.info(f"搜索完成，找到 {len(results)} 个结果，耗时 {elapsed_time:.2f} 秒")

            # 保存搜索历史记录
            if save_to_history and self.enable_history and self.history_manager:
                try:
                    # 计算最高相似度
                    top_similarity = max([r['similarity'] for r in results]) if results else 0.0

                    # 获取实际使用的模型名称
                    actual_model_name = self.model_name or self.feature_extractor.get_model_name()

                    # 根据是否复用图像选择不同的保存方法
                    if reuse_image_path:
                        # 复用现有图像
                        record_id = self.history_manager.add_search_record_with_existing_image(
                            existing_image_path=reuse_image_path,
                            target_folders=valid_paths,
                            model_type=self.model_type,
                            model_name=actual_model_name,
                            similarity_threshold=similarity_threshold,
                            results_count=len(results),
                            execution_time=elapsed_time,
                            max_results=top_n,
                            top_similarity=top_similarity,
                            search_results=results
                        )
                        logger.info(f"搜索历史已保存（复用图像），记录ID: {record_id}")
                    else:
                        # 保存新图像
                        record_id = self.history_manager.add_search_record(
                            query_image_path=query_image_path,
                            target_folders=valid_paths,
                            model_type=self.model_type,
                            model_name=actual_model_name,
                            similarity_threshold=similarity_threshold,
                            results_count=len(results),
                            execution_time=elapsed_time,
                            max_results=top_n,
                            top_similarity=top_similarity,
                            search_results=results
                        )
                        logger.info(f"搜索历史已保存，记录ID: {record_id}")
                except Exception as e:
                    logger.warning(f"保存搜索历史失败: {e}")

            if progress_callback:
                progress_callback(1.0)  # 完成

            return results
            
        except Exception as e:
            logger.error(f"搜索过程中发生错误: {e}")
            raise 