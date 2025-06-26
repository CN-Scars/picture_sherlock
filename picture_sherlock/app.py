"""
图像相似度搜索应用
主应用程序入口，使用Streamlit构建用户界面
"""
import os
import sys
import time
import tempfile
import shutil
import glob
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

# 添加导入模块的路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import streamlit as st
import pandas as pd
import numpy as np
import torch  # 导入torch用于硬件检测
from PIL import Image
import plotly.express as px

# 优先使用绝对导入
try:
    # 尝试绝对导入（当安装为包或添加到sys.path时）
    from picture_sherlock.utils import logger, validate_paths, CACHE_DIR
    from picture_sherlock.similarity_search import SimilaritySearch
    from picture_sherlock.feature_extractor import DEFAULT_MODEL_CONFIG
    from picture_sherlock.file_selector import folium_file_explorer
    from picture_sherlock.i18n import _, init_language, change_language, SUPPORTED_LANGUAGES
    from picture_sherlock.cache_manager import CacheManager
    from picture_sherlock.search_history_manager import SearchHistoryManager
except ImportError:
    # 尝试直接导入（当在目录内运行时）
    from utils import logger, validate_paths, CACHE_DIR
    from similarity_search import SimilaritySearch
    from feature_extractor import DEFAULT_MODEL_CONFIG
    from file_selector import folium_file_explorer
    from i18n import _, init_language, change_language, SUPPORTED_LANGUAGES
    from cache_manager import CacheManager
    from search_history_manager import SearchHistoryManager

# 检测是否安装了streamlit-folium
try:
    import folium
    from streamlit_folium import st_folium
    STREAMLIT_FOLIUM_AVAILABLE = True
except ImportError:
    STREAMLIT_FOLIUM_AVAILABLE = False

# Hugging Face模型缓存目录
HUGGINGFACE_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

# GitHub仓库链接
GITHUB_REPO_URL = "https://github.com/yourusername/picture-sherlock"

# 初始化语言设置
init_language()

# 配置页面
st.set_page_config(
    page_title=_("app_title"),
    page_icon=_("app_icon"),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': _("about_content").replace("https://github.com/yourusername/picture-sherlock", GITHUB_REPO_URL)
    }
)

# 定义应用样式
st.markdown("""
<style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .preview-img {
        width: 100%;
        border-radius: 5px;
    }
    .result-img {
        width: 100%;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .result-img:hover {
        transform: scale(1.05);
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .sidebar .stButton button {
        width: 100%;
    }
    .download-progress {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
    .model-info {
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #555;
    }
    .folder-selector {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .folder-list {
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        max-height: 200px;
        overflow-y: auto;
    }
    .folder-item {
        padding: 0.3rem 0.5rem;
        margin-bottom: 0.3rem;
        background-color: white;
        border-radius: 0.3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .search-button {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .cache-info {
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #555;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
    .language-selector {
        margin-top: 1rem;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    加载图像文件

    Args:
        image_path: 图像文件路径

    Returns:
        Optional[Image.Image]: 加载的图像对象，如果加载失败则返回None
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"加载图像失败: {image_path}, 错误: {e}")
        return None


def get_file_download_link(file_path: str) -> str:
    """
    创建文件下载链接

    Args:
        file_path: 文件路径

    Returns:
        str: HTML格式的下载链接
    """
    try:
        # 获取文件名
        file_name = os.path.basename(file_path)

        # 创建下载链接
        return f'<a href="file:///{file_path}" target="_blank">{file_name}</a>'
    except Exception as e:
        logger.error(f"创建下载链接失败: {file_path}, 错误: {e}")
        return file_path


@st.cache_resource
def get_similarity_search(model_type: str, model_name: Optional[str] = None, _progress_callback: Optional[Callable] = None) -> SimilaritySearch:
    """
    获取相似度搜索实例
    
    Args:
        model_type: 模型类型
        model_name: 模型名称
        _progress_callback: 进度回调函数
        
    Returns:
        SimilaritySearch: 相似度搜索实例
    """
    return SimilaritySearch(
        model_type=model_type,
        model_name=model_name,
        download_progress_callback=_progress_callback
    )


def clear_index_cache() -> Tuple[bool, str]:
    """
    清除索引缓存文件
    
    Returns:
        Tuple[bool, str]: 是否成功删除和消息
    """
    try:
        # 使用缓存管理器清除所有缓存
        cache_manager = CacheManager()
        deleted_count = cache_manager.clear_all_caches()
        
        if deleted_count == 0:
            return False, _("no_index_cache_found")
            
        return True, _("index_cache_deleted").format(count=deleted_count)
        
    except Exception as e:
        logger.error(f"清除索引缓存出错: {e}")
        return False, _("index_cache_delete_error").format(error=str(e))


def clear_model_cache() -> Tuple[bool, str]:
    """
    清除Hugging Face模型缓存
    
    Returns:
        Tuple[bool, str]: 是否成功删除和消息
    """
    try:
        # 检查Hugging Face缓存目录是否存在
        if not os.path.exists(HUGGINGFACE_CACHE_DIR):
            return False, _("no_model_cache_found")
            
        # 获取模型缓存文件夹大小
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(HUGGINGFACE_CACHE_DIR):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        
        # 转换为MB
        size_mb = total_size / (1024 * 1024)
        
        # 删除Hugging Face缓存目录
        shutil.rmtree(HUGGINGFACE_CACHE_DIR)
        
        # 重新创建空目录
        os.makedirs(HUGGINGFACE_CACHE_DIR, exist_ok=True)
        
        return True, _("model_cache_deleted").format(size=f"{size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"清除模型缓存出错: {e}")
        return False, _("model_cache_delete_error").format(error=str(e))


def get_cache_info() -> Dict[str, Any]:
    """
    获取缓存信息
    
    Returns:
        Dict[str, Any]: 缓存信息
    """
    try:
        # 使用缓存管理器获取缓存信息
        cache_manager = CacheManager()
        cache_info = cache_manager.get_cache_info()
        
        # 获取模型缓存大小
        model_cache_size = 0
        if os.path.exists(HUGGINGFACE_CACHE_DIR):
            for dirpath, dirnames, filenames in os.walk(HUGGINGFACE_CACHE_DIR):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    model_cache_size += os.path.getsize(file_path)
        
        # 添加模型缓存信息
        cache_info['model_cache_size_bytes'] = model_cache_size
        cache_info['model_cache_size_mb'] = model_cache_size / (1024 * 1024)
        
        return cache_info

    except Exception as e:
        logger.error(f"获取缓存信息出错: {e}")
        return {
            'total_caches': 0,
            'total_size_bytes': 0,
            'total_size_mb': 0,
            'model_counts': {},
            'directory_stats': {},
            'status_counts': {},
            'model_cache_size_bytes': 0,
            'model_cache_size_mb': 0,
            'error': str(e)
        }


def delete_cache(cache_id: str) -> bool:
    """
    删除指定的缓存
    
    Args:
        cache_id: 缓存ID
        
    Returns:
        bool: 删除是否成功
    """
    try:
        cache_manager = CacheManager()
        return cache_manager.delete_cache(cache_id)
    except Exception as e:
        logger.error(f"删除缓存出错: {e}")
        return False


def rebuild_cache(cache_id: str) -> bool:
    """
    重建指定的缓存
    
    Args:
        cache_id: 缓存ID
        
    Returns:
        bool: 重建是否成功
    """
    try:
        # 获取缓存信息
        cache_manager = CacheManager()
        caches = cache_manager.get_all_caches()
        cache_entry = next((c for c in caches if c['id'] == cache_id), None)
        
        if not cache_entry:
            return False
            
        # 创建搜索实例并重建索引
        model_name = cache_entry['model']
        dir_paths = cache_entry['dir']
        
        # 更新缓存状态
        cache_manager.update_cache_status(cache_id, 'rebuilding')
        
        # 创建搜索实例
        search = SimilaritySearch(model_type=get_model_type_from_name(model_name))
        
        # 重建索引
        success = search.build_index(dir_paths, force_rebuild=True)
        
        # 更新缓存状态
        if success:
            cache_manager.update_cache_status(cache_id, 'indexed')
        else:
            cache_manager.update_cache_status(cache_id, 'failed')
            
        return success
    except Exception as e:
        logger.error(f"重建缓存出错: {e}")
        return False


def batch_delete_caches(cache_ids: List[str]) -> Tuple[int, int]:
    """
    批量删除缓存
    
    Args:
        cache_ids: 缓存ID列表
        
    Returns:
        Tuple[int, int]: (成功删除数量, 失败删除数量)
    """
    try:
        cache_manager = CacheManager()
        return cache_manager.batch_delete_caches(cache_ids)
    except Exception as e:
        logger.error(f"批量删除缓存出错: {e}")
        return (0, len(cache_ids))


def batch_rebuild_caches(cache_ids: List[str]) -> Tuple[int, int]:
    """
    批量重建缓存
    
    Args:
        cache_ids: 缓存ID列表
        
    Returns:
        Tuple[int, int]: (成功重建数量, 失败重建数量)
    """
    success_count = 0
    fail_count = 0
    
    for cache_id in cache_ids:
        if rebuild_cache(cache_id):
            success_count += 1
        else:
            fail_count += 1
            
    return (success_count, fail_count)


def get_model_type_from_name(model_name: str) -> str:
    """
    根据模型名称获取模型类型
    
    Args:
        model_name: 模型名称
        
    Returns:
        str: 模型类型
    """
    if 'clip' in model_name.lower():
        return 'clip'
    elif 'resnet' in model_name.lower():
        return 'resnet'
    else:
        return 'clip'  # 默认使用CLIP


def format_datetime(datetime_str: str) -> str:
    """
    格式化日期时间字符串
    
    Args:
        datetime_str: ISO格式的日期时间字符串
        
    Returns:
        str: 格式化后的日期时间字符串
    """
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(datetime_str)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return datetime_str


def format_directory_path(dir_path: str) -> str:
    """
    格式化目录路径，截断长路径
    
    Args:
        dir_path: 目录路径
        
    Returns:
        str: 格式化后的路径
    """
    max_length = 30
    if len(dir_path) <= max_length:
        return dir_path
        
    # 截断中间部分
    parts = dir_path.split(os.sep)
    if len(parts) <= 2:
        return dir_path
        
    return os.path.join(parts[0], '...', parts[-1])


def show_search_history():
    """
    显示搜索历史界面
    """
    st.header(_("search_history"))

    # 初始化搜索历史管理器
    try:
        history_manager = SearchHistoryManager()
    except Exception as e:
        st.error(f"Failed to initialize search history manager: {e}")
        return

    # 获取筛选参数
    model_filter = st.session_state.get("history_model_filter", "all")
    date_filter = st.session_state.get("history_date_filter", None)
    favorites_only = st.session_state.get("history_favorites_only", False)
    search_term = st.session_state.get("history_search_term", "")

    # 构建筛选条件
    filter_kwargs = {}
    if model_filter != "all":
        filter_kwargs["model_type"] = model_filter
    if date_filter:
        from datetime import datetime, time
        filter_kwargs["date_from"] = datetime.combine(date_filter, time.min)
        filter_kwargs["date_to"] = datetime.combine(date_filter, time.max)
    if favorites_only:
        filter_kwargs["favorites_only"] = True

    # 获取搜索历史记录
    records = history_manager.get_records_by_filter(**filter_kwargs)

    # 如果有搜索词，进一步筛选
    if search_term:
        filtered_records = []
        for record in records:
            # 在标签、备注中搜索
            search_fields = [
                " ".join(record.get("user_data", {}).get("tags", [])),
                record.get("user_data", {}).get("notes", "")
            ]
            if any(search_term.lower() in field.lower() for field in search_fields):
                filtered_records.append(record)
        records = filtered_records

    # 显示统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(_("total_searches"), len(records))
    with col2:
        favorite_count = sum(1 for r in records if r.get("user_data", {}).get("favorite", False))
        st.metric(_("favorite_count"), favorite_count)
    with col3:
        if records:
            avg_time = sum(r.get("results", {}).get("execution_time", 0) for r in records) / len(records)
            st.metric(_("average_execution_time"), f"{avg_time:.2f}s")
        else:
            st.metric(_("average_execution_time"), "0.00s")

    st.markdown("---")

    # 显示搜索历史记录
    if not records:
        st.info(_("no_search_history"))
        return

    # 分页设置
    records_per_page = 10
    total_pages = (len(records) + records_per_page - 1) // records_per_page

    if "history_current_page" not in st.session_state:
        st.session_state.history_current_page = 0

    # 分页控制
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        with col1:
            if st.button(_("previous_page"), disabled=st.session_state.history_current_page == 0):
                st.session_state.history_current_page -= 1
                st.rerun()
        with col2:
            st.write(f"{st.session_state.history_current_page + 1}/{total_pages}")
        with col3:
            page_input = st.number_input(
                _("go_to_page"),
                min_value=1,
                max_value=total_pages,
                value=st.session_state.history_current_page + 1,
                key="page_input"
            )
            if page_input != st.session_state.history_current_page + 1:
                st.session_state.history_current_page = page_input - 1
                st.rerun()
        with col4:
            st.write(f"{len(records)} {_('search_history_count').format(count='')}")
        with col5:
            if st.button(_("next_page"), disabled=st.session_state.history_current_page >= total_pages - 1):
                st.session_state.history_current_page += 1
                st.rerun()

    # 获取当前页的记录
    start_idx = st.session_state.history_current_page * records_per_page
    end_idx = start_idx + records_per_page
    page_records = records[start_idx:end_idx]

    # 显示记录
    for i, record in enumerate(page_records):
        # 生成更有意义的标题
        timestamp = record.get('timestamp', '')[:19]
        model_type = record.get('search_config', {}).get('model_type', 'unknown').upper()
        results_count = record.get('results', {}).get('count', 0)
        target_folders = record.get('search_config', {}).get('target_folders', [])
        folder_name = os.path.basename(target_folders[0]) if target_folders else _("unknown_folder")

        # 构建标题：时间 - 模型 - 结果数 - 文件夹
        title = f"🔍 {timestamp} | {model_type} | {results_count}{_('results_count_suffix')} | {folder_name}"

        with st.expander(title):
            show_search_record_details(record, history_manager)

    # 处理对话框
    handle_history_dialogs(history_manager)


def show_search_record_details(record, history_manager):
    """
    显示搜索记录详情
    """
    col1, col2 = st.columns([2, 1])

    with col1:
        # 显示查询图像
        record_id = record.get('id')
        stored_image_path = record.get('query_image', {}).get('stored_path')

        if history_manager:
            # 优先使用stored_path，如果没有则尝试通过record_id查找
            if stored_image_path:
                # 将相对路径转换为绝对路径
                if not os.path.isabs(stored_image_path):
                    query_image_path = os.path.join(history_manager.storage_dir, stored_image_path)
                else:
                    query_image_path = stored_image_path
            else:
                # 旧记录：尝试通过record_id查找
                query_image_path = history_manager.get_query_image_path(record_id)

            if query_image_path and os.path.exists(query_image_path):
                try:
                    query_image = load_image(query_image_path)
                    if query_image:
                        st.markdown(f"**{_('query_image')}:**")
                        st.image(query_image, width=200, caption=_("query_image_caption"))
                    else:
                        st.markdown(f"**{_('query_image')}:** {_('query_image_loading_failed')}")
                except Exception as e:
                    st.markdown(f"**{_('query_image')}:** {_('query_image_loading_error')} - {str(e)[:50]}")
            else:
                st.markdown(f"**{_('query_image')}:** {_('query_image_file_not_found')}")
        else:
            st.markdown(f"**{_('query_image')}:** {_('query_image_no_storage')}")

        st.markdown("---")

        # 基本信息
        st.markdown(f"**{_('search_time')}:** {record.get('timestamp', '')[:19]}")
        st.markdown(f"**{_('model_used')}:** {record.get('search_config', {}).get('model_name', 'Unknown')}")
        st.markdown(f"**{_('results_found')}:** {record.get('results', {}).get('count', 0)}")
        st.markdown(f"**{_('execution_time')}:** {record.get('results', {}).get('execution_time', 0):.2f}s")

        # 目标文件夹
        folders = record.get('search_config', {}).get('target_folders', [])
        if folders:
            st.markdown(f"**{_('target_folders')}:**")
            for folder in folders:
                st.markdown(f"  • {folder}")

        # 搜索结果预览
        similar_images = record.get('results', {}).get('similar_images', [])
        if similar_images:
            st.markdown(f"**{_('search_results_preview', count=5)}:**")

            # 显示前5个结果的缩略图
            cols = st.columns(5)
            for i, result in enumerate(similar_images[:5]):
                with cols[i]:
                    try:
                        image_path = result.get('path', '')
                        similarity = result.get('similarity', 0)

                        if os.path.exists(image_path):
                            image = load_image(image_path)
                            if image:
                                st.image(
                                    image,
                                    caption=f"{similarity:.3f}",
                                    use_container_width=True
                                )
                            else:
                                st.text(f"{_('cannot_load_image')}\n{os.path.basename(image_path)}")
                        else:
                            st.text(f"{_('file_not_exists')}\n{os.path.basename(image_path)}")
                    except Exception as e:
                        st.text(f"{_('error_short')}\n{str(e)[:20]}")

            # 显示完整结果列表
            if len(similar_images) > 5:
                st.markdown("---")
                if st.button(f"📋 {_('view_all_results', count=len(similar_images))}", key=f"show_all_{record.get('id', 'unknown')}"):
                    st.session_state[f"show_all_results_{record.get('id', 'unknown')}"] = True

                if st.session_state.get(f"show_all_results_{record.get('id', 'unknown')}", False):
                    st.markdown(f"**{_('complete_results_list', count=len(similar_images))}:**")
                    for i, result in enumerate(similar_images):
                        image_path = result.get('path', '')
                        similarity = result.get('similarity', 0)
                        st.markdown(f"{i+1}. **{os.path.basename(image_path)}** - {_('similarity_label')}: {similarity:.4f}")
                        st.markdown(f"   {_('path_label')}: `{image_path}`")

                    if st.button(f"🔼 {_('collapse_results')}", key=f"hide_all_{record.get('id', 'unknown')}"):
                        st.session_state[f"show_all_results_{record.get('id', 'unknown')}"] = False
                        st.rerun()
        else:
            st.markdown(f"**{_('search_results_no_data')}**")

    with col2:
        # 操作按钮
        record_id = record.get('id')
        is_favorite = record.get('user_data', {}).get('favorite', False)

        # 收藏/取消收藏
        if st.button(
            _("remove_from_favorites") if is_favorite else _("add_to_favorites"),
            key=f"fav_{record_id}"
        ):
            history_manager.update_record(record_id, favorite=not is_favorite)
            st.rerun()

        # 重新执行搜索（跳转到搜索页面执行）
        if st.button(_("repeat_search"), key=f"repeat_{record_id}"):
            # 获取历史记录中的搜索参数
            target_folders = record.get('search_config', {}).get('target_folders', [])
            model_name = record.get('search_config', {}).get('model_name', '')
            model_type = record.get('search_config', {}).get('model_type', '')
            stored_image_path = record.get('query_image', {}).get('stored_path', '')
            max_results = record.get('search_config', {}).get('max_results', 10)

            # 将相对路径转换为绝对路径
            if stored_image_path:
                # 如果是相对路径，需要基于历史管理器的存储目录构建完整路径
                if not os.path.isabs(stored_image_path):
                    full_image_path = os.path.join(history_manager.storage_dir, stored_image_path)
                else:
                    full_image_path = stored_image_path
            else:
                full_image_path = ''

            # 检查必要的数据是否完整
            missing_data = []
            if not target_folders:
                missing_data.append(_("target_folders"))
            if not model_name:
                missing_data.append(_("model_name"))
            if not model_type:
                missing_data.append(_("model_type"))
            if not full_image_path or not os.path.exists(full_image_path):
                missing_data.append(_("query_image"))

            if missing_data:
                st.error(_("missing_search_data").format(data=", ".join(missing_data)))
                return

            # 设置搜索参数并跳转到搜索页面
            st.session_state.selected_folders = target_folders
            st.session_state.repeat_search_model_name = model_name
            st.session_state.repeat_search_model_type = model_type
            st.session_state.repeat_search_image_path = full_image_path
            st.session_state.repeat_search_max_results = max_results
            st.session_state.repeat_search_stored_path = stored_image_path  # 保存原始相对路径用于历史记录
            st.session_state.switch_to_search_tab = True
            st.session_state.auto_execute_search = True  # 标记自动执行搜索
            st.success(_("switching_to_search_tab"))
            st.rerun()

        # 删除记录
        if st.button(_("delete_record"), key=f"del_{record_id}", type="secondary"):
            if st.session_state.get(f"confirm_delete_{record_id}", False):
                history_manager.delete_record(record_id)
                st.success("Record deleted")
                st.rerun()
            else:
                st.session_state[f"confirm_delete_{record_id}"] = True
                st.warning(_("delete_record_confirm"))

    # 用户数据编辑
    st.markdown("---")
    st.markdown(f"**{_('edit_tags')} & {_('edit_notes')}**")

    current_tags = record.get('user_data', {}).get('tags', [])
    current_notes = record.get('user_data', {}).get('notes', '')

    col1, col2 = st.columns(2)
    with col1:
        new_tags_str = st.text_input(
            _("user_tags"),
            value=", ".join(current_tags),
            placeholder=_("tags_placeholder"),
            key=f"tags_{record_id}"
        )

    with col2:
        new_notes = st.text_area(
            _("user_notes"),
            value=current_notes,
            placeholder=_("notes_placeholder"),
            key=f"notes_{record_id}",
            height=100
        )

    if st.button(_("save_changes"), key=f"save_{record_id}"):
        new_tags = [tag.strip() for tag in new_tags_str.split(",") if tag.strip()]
        history_manager.update_record(
            record_id,
            tags=new_tags,
            notes=new_notes
        )
        st.success("Changes saved")
        st.rerun()


def handle_history_dialogs(history_manager):
    """
    处理搜索历史相关的对话框
    """
    # 导出对话框
    if st.session_state.get("show_export_dialog", False):
        with st.form("export_form"):
            st.subheader(_("export_history"))
            export_path = st.text_input("Export file path", value="search_history_export.json")
            include_user_data = st.checkbox("Include user data (tags, notes)", value=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button(_("export_history")):
                    try:
                        if history_manager.export_history(export_path, include_user_data):
                            st.success(_("export_success"))
                        else:
                            st.error(_("export_failed"))
                    except Exception as e:
                        st.error(f"{_('export_failed')}: {e}")
                    st.session_state.show_export_dialog = False
                    st.rerun()
            with col2:
                if st.form_submit_button(_("cancel_changes")):
                    st.session_state.show_export_dialog = False
                    st.rerun()

    # 导入对话框
    if st.session_state.get("show_import_dialog", False):
        with st.form("import_form"):
            st.subheader(_("import_history"))
            uploaded_file = st.file_uploader("Choose history file", type=['json'])
            merge_data = st.checkbox("Merge with existing data", value=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button(_("import_history")):
                    if uploaded_file:
                        try:
                            # 保存上传的文件到临时位置
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue().decode())
                                tmp_path = tmp_file.name

                            if history_manager.import_history(tmp_path, merge_data):
                                st.success(_("import_success"))
                            else:
                                st.error(_("import_failed"))

                            # 清理临时文件
                            os.unlink(tmp_path)
                        except Exception as e:
                            st.error(f"{_('import_failed')}: {e}")
                    else:
                        st.warning("Please select a file to import")
                    st.session_state.show_import_dialog = False
                    st.rerun()
            with col2:
                if st.form_submit_button(_("cancel_changes")):
                    st.session_state.show_import_dialog = False
                    st.rerun()

    # 清空历史记录对话框
    if st.session_state.get("show_clear_history_dialog", False):
        with st.form("clear_history_form"):
            st.subheader(_("clear_all_history"))
            st.warning(_("clear_history_confirm"))

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button(_("clear_all_history"), type="primary"):
                    try:
                        if history_manager.clear_all_records():
                            st.success(_("history_cleared"))
                        else:
                            st.error("Failed to clear history")
                    except Exception as e:
                        st.error(f"Failed to clear history: {e}")
                    st.session_state.show_clear_history_dialog = False
                    st.rerun()
            with col2:
                if st.form_submit_button(_("cancel_changes")):
                    st.session_state.show_clear_history_dialog = False
                    st.rerun()

    # 清理孤儿图像对话框
    if st.session_state.get("show_cleanup_images_dialog", False):
        with st.form("cleanup_images_form"):
            st.subheader(f"🧹 {_('cleanup_images_title')}")
            st.info(_("cleanup_images_description"))
            st.warning(f"⚠️ {_('cleanup_images_warning')}")

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button(_("start_cleanup"), type="primary"):
                    try:
                        cleaned_count = history_manager.cleanup_orphaned_images()
                        if cleaned_count > 0:
                            st.success(_("cleanup_success").format(count=cleaned_count))
                        else:
                            st.info(_("cleanup_no_files"))
                        st.session_state.show_cleanup_images_dialog = False
                        st.rerun()
                    except Exception as e:
                        st.error(_("cleanup_failed").format(error=str(e)))

            with col2:
                if st.form_submit_button(_("cancel")):
                    st.session_state.show_cleanup_images_dialog = False
                    st.rerun()


def show_favorites_management():
    """
    显示收藏管理界面
    """
    st.header(f"🌟 {_('favorites_management')}")

    # 初始化搜索历史管理器
    try:
        history_manager = SearchHistoryManager()
    except Exception as e:
        st.error(f"Failed to initialize search history manager: {e}")
        return

    # 获取筛选参数
    sort_by = st.session_state.get("favorites_sort_by", "newest")
    model_filter = st.session_state.get("favorites_model_filter", "all")

    # 构建筛选条件
    filter_kwargs = {"favorites_only": True}
    if model_filter != "all":
        filter_kwargs["model_type"] = model_filter

    # 获取收藏记录
    favorite_records = history_manager.get_records_by_filter(**filter_kwargs)

    # 排序
    if sort_by == "newest":
        favorite_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    elif sort_by == "oldest":
        favorite_records.sort(key=lambda x: x.get('timestamp', ''))
    elif sort_by == "most_similar":
        favorite_records.sort(key=lambda x: x.get('results', {}).get('top_similarity', 0), reverse=True)
    elif sort_by == "execution_time":
        favorite_records.sort(key=lambda x: x.get('results', {}).get('execution_time', 0))

    # 显示统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(_("favorites_total"), len(favorite_records))
    with col2:
        if favorite_records:
            avg_similarity = sum(r.get('results', {}).get('top_similarity', 0) for r in favorite_records) / len(favorite_records)
            st.metric(_("average_similarity"), f"{avg_similarity:.3f}")
    with col3:
        if favorite_records:
            avg_time = sum(r.get('results', {}).get('execution_time', 0) for r in favorite_records) / len(favorite_records)
            st.metric(_("average_execution_time"), f"{avg_time:.2f}s")

    if not favorite_records:
        st.info(f"🌟 {_('no_favorites')}")
        return

    st.markdown("---")

    # 分页显示
    items_per_page = 5
    total_pages = (len(favorite_records) + items_per_page - 1) // items_per_page

    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.selectbox(
                "页面",
                range(1, total_pages + 1),
                format_func=lambda x: f"第 {x} 页 / 共 {total_pages} 页"
            ) - 1
    else:
        current_page = 0

    # 获取当前页的记录
    start_idx = current_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(favorite_records))
    page_records = favorite_records[start_idx:end_idx]

    # 显示收藏记录
    for i, record in enumerate(page_records):
        # 生成标题
        timestamp = record.get('timestamp', '')[:19]
        model_type = record.get('search_config', {}).get('model_type', 'unknown').upper()
        results_count = record.get('results', {}).get('count', 0)
        top_similarity = record.get('results', {}).get('top_similarity', 0)
        target_folders = record.get('search_config', {}).get('target_folders', [])
        folder_name = os.path.basename(target_folders[0]) if target_folders else _("unknown_folder")

        # 构建标题：时间 - 模型 - 结果数 - 最高相似度 - 文件夹
        title = f"⭐ {timestamp} | {model_type} | {results_count}{_('results_count_suffix')} | {top_similarity:.3f} | {folder_name}"

        with st.expander(title):
            show_search_record_details(record, history_manager)

    # 处理对话框
    handle_favorites_dialogs(history_manager)


def handle_favorites_dialogs(history_manager):
    """
    处理收藏管理相关的对话框
    """
    # 取消全部收藏对话框
    if st.session_state.get("show_unfavorite_all_dialog", False):
        with st.form("unfavorite_all_form"):
            st.warning(f"⚠️ {_('unfavorite_all_confirm')}")
            st.markdown(_("unfavorite_all_description"))

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button(_("unfavorite_all"), type="primary"):
                    try:
                        # 获取所有收藏记录
                        favorite_records = history_manager.get_records_by_filter(favorites_only=True)

                        # 取消所有收藏
                        for record in favorite_records:
                            history_manager.update_record(record['id'], favorite=False)

                        st.success(_("unfavorite_all_success").format(count=len(favorite_records)))
                        st.session_state.show_unfavorite_all_dialog = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"{_('unfavorite_failed')}: {e}")

            with col2:
                if st.form_submit_button(_("cancel")):
                    st.session_state.show_unfavorite_all_dialog = False
                    st.rerun()


def show_tags_management():
    """
    显示标签管理界面
    """
    st.header(f"🏷️ {_('tags_management')}")

    # 初始化搜索历史管理器
    try:
        history_manager = SearchHistoryManager()
    except Exception as e:
        st.error(f"Failed to initialize search history manager: {e}")
        return

    # 获取所有标签
    all_tags = history_manager.get_all_tags()

    if not all_tags:
        st.info(f"📝 {_('no_tags_yet')}")
        return

    # 标签统计概览
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(_("total_tags"), len(all_tags))
    with col2:
        total_usage = sum(all_tags.values())
        st.metric(_("total_tag_usage"), total_usage)
    with col3:
        avg_usage = total_usage / len(all_tags) if all_tags else 0
        st.metric(_("average_tag_usage"), f"{avg_usage:.1f}")

    st.markdown("---")

    # 标签云显示
    st.subheader(f"🌤️ {_('tag_cloud')}")

    # 创建标签云数据
    tag_cloud_data = []
    max_usage = max(all_tags.values()) if all_tags else 1

    for tag, count in sorted(all_tags.items(), key=lambda x: x[1], reverse=True):
        # 根据使用频率计算字体大小
        size = max(12, min(32, int(12 + (count / max_usage) * 20)))
        tag_cloud_data.append({
            "tag": tag,
            "count": count,
            "size": size
        })

    # 显示标签云（使用表格形式，因为streamlit没有原生标签云组件）
    st.markdown(f"### {_('tag_statistics')}")

    # 创建多列布局显示标签
    cols = st.columns(4)
    for i, tag_data in enumerate(tag_cloud_data):
        col_idx = i % 4
        with cols[col_idx]:
            tag = tag_data["tag"]
            count = tag_data["count"]

            # 使用按钮显示标签，点击可以查看相关记录
            if st.button(f"🏷️ {tag} ({count})", key=f"tag_button_{tag}"):
                st.session_state.selected_tag_for_view = tag
                st.rerun()

    st.markdown("---")

    # 标签详细统计表格
    st.subheader(f"📊 {_('tag_detailed_stats')}")

    # 创建DataFrame
    df = pd.DataFrame([
        {_("tag_name"): tag, _("tag_usage_count"): count, _("tag_usage_rate"): f"{count/total_usage*100:.1f}%"}
        for tag, count in sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
    ])

    st.dataframe(df, hide_index=True, use_container_width=True)

    # 如果选择了标签，显示相关记录
    if st.session_state.get("selected_tag_for_view"):
        selected_tag = st.session_state.selected_tag_for_view
        st.markdown("---")
        st.subheader(f"🔍 {_('tag_related_records', tag=selected_tag)}")

        # 获取该标签的所有记录
        tagged_records = history_manager.get_records_by_tag(selected_tag)

        if tagged_records:
            st.info(_('tag_records_found', count=len(tagged_records), tag=selected_tag))

            # 显示记录
            for i, record in enumerate(tagged_records[:10]):  # 只显示前10条
                timestamp = record.get('timestamp', '')[:19]
                model_type = record.get('search_config', {}).get('model_type', 'unknown').upper()
                results_count = record.get('results', {}).get('count', 0)

                title = f"🔍 {timestamp} | {model_type} | {results_count} {_('results_count_suffix')}"

                with st.expander(title):
                    show_search_record_details(record, history_manager)

            if len(tagged_records) > 10:
                st.info(_('more_records_hidden', count=len(tagged_records) - 10))
        else:
            st.warning(_('tag_no_records', tag=selected_tag))

        # 清除选择按钮
        if st.button(f"🔙 {_('back_to_tag_list')}"):
            st.session_state.selected_tag_for_view = None
            st.rerun()


def show_cache_management():
    """
    显示缓存管理界面
    """
    st.header(_("cache_management_title"))
    
    # 初始化缓存管理器
    cache_manager = CacheManager()
    
    # 获取缓存信息
    cache_info = get_cache_info()
    
    # 显示缓存统计信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(_("index_cache_info"))
        st.info(_(
            "index_cache_stats").format(
            count=cache_info['total_caches'],
            size=f"{cache_info['total_size_mb']:.2f} MB"
        ))
        
        # 显示清除索引缓存按钮
        if st.button(_("clear_index_cache_button"), key="clear_index_cache"):
            success, message = clear_index_cache()
            if success:
                st.success(message)
            else:
                st.warning(message)
                
    with col2:
        st.subheader(_("model_cache_info"))
        st.info(_(
            "model_cache_stats").format(
            size=f"{cache_info.get('model_cache_size_mb', 0):.2f} MB"
        ))
        
        # 显示清除模型缓存按钮
        if st.button(_("clear_model_cache_button"), key="clear_model_cache"):
            success, message = clear_model_cache()
            if success:
                st.success(message)
            else:
                st.warning(message)
    
    # 显示模型使用统计
    if cache_info['model_counts']:
        st.subheader(_("model_usage_stats"))
        
        # 准备图表数据
        models = list(cache_info['model_counts'].keys())
        counts = list(cache_info['model_counts'].values())
        
        # 创建条形图
        fig = px.bar(
            x=models, 
            y=counts,
            labels={'x': _("model_name"), 'y': _("cache_count")},
            title=_("model_usage_chart_title")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 显示缓存列表
    st.subheader(_("cache_list_title"))
    
    # 初始化会话状态变量
    if 'selected_caches' not in st.session_state:
        st.session_state.selected_caches = []
    if 'search_keyword' not in st.session_state:
        st.session_state.search_keyword = ""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    # 搜索框
    search_keyword = st.text_input(_("search_cache_placeholder"), value=st.session_state.search_keyword, key="cache_search_input")
    if search_keyword != st.session_state.search_keyword:
        st.session_state.search_keyword = search_keyword
        st.session_state.current_page = 0
    
    # 获取缓存列表
    caches = cache_manager.search_caches(search_keyword)
    
    # 批量操作按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button(_("batch_rebuild_button"), key="batch_rebuild", disabled=len(st.session_state.selected_caches) == 0):
            success_count, fail_count = batch_rebuild_caches(st.session_state.selected_caches)
            st.success(_("batch_rebuild_result").format(success=success_count, fail=fail_count))
            st.session_state.selected_caches = []
            st.experimental_rerun()
    
    with col2:
        if st.button(_("batch_delete_button"), key="batch_delete", disabled=len(st.session_state.selected_caches) == 0):
            success_count, fail_count = batch_delete_caches(st.session_state.selected_caches)
            st.success(_("batch_delete_result").format(success=success_count, fail=fail_count))
            st.session_state.selected_caches = []
            st.experimental_rerun()

    # 分页
    items_per_page = 10
    total_pages = (len(caches) - 1) // items_per_page + 1 if caches else 0
    
    if total_pages > 0:
        page_start = st.session_state.current_page * items_per_page
        page_end = min(page_start + items_per_page, len(caches))
        
        # 显示页码信息
        st.text(_("pagination_info").format(
            start=page_start + 1,
            end=page_end,
            total=len(caches),
            page=st.session_state.current_page + 1,
            total_pages=total_pages
        ))
        
        # 分页导航
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("◀", disabled=st.session_state.current_page == 0):
                st.session_state.current_page -= 1
                st.experimental_rerun()
        
        with col3:
            if st.button("▶", disabled=st.session_state.current_page >= total_pages - 1):
                st.session_state.current_page += 1
                st.experimental_rerun()
        
        # 显示当前页的缓存
        current_page_caches = caches[page_start:page_end]
        
        # 创建缓存表格
        for cache in current_page_caches:
            cache_id = cache['id']
            model_name = cache['model']
            dirs = cache['dir'] if isinstance(cache['dir'], list) else [cache['dir']]
            created_at = format_datetime(cache['created_at'])
            updated_at = format_datetime(cache.get('updated_at', cache['created_at']))
            status = cache.get('status', 'created')
            
            # 计算缓存大小
            cache_size = cache_manager.get_cache_size(cache_id)
            size_mb = cache_size / (1024 * 1024)
            
            # 创建缓存项容器
            with st.container():
                col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1])
                
                # 选择框
                with col1:
                    is_selected = cache_id in st.session_state.selected_caches
                    if st.checkbox("", value=is_selected, key=f"select_{cache_id}"):
                        if cache_id not in st.session_state.selected_caches:
                            st.session_state.selected_caches.append(cache_id)
                    else:
                        if cache_id in st.session_state.selected_caches:
                            st.session_state.selected_caches.remove(cache_id)
                
                # 缓存信息
                with col2:
                    st.markdown(f"**{model_name}** ({cache_id[:8]}...)")
                    st.text(f"{_('cache_dirs')}: {', '.join([format_directory_path(d) for d in dirs])}")
                    st.text(f"{_('cache_created')}: {created_at}")
                    st.text(f"{_('cache_updated')}: {updated_at}")
                    st.text(f"{_('cache_status')}: {status}")
                    st.text(f"{_('cache_size')}: {size_mb:.2f} MB")
                
                # 重建按钮
                with col3:
                    if st.button(_("rebuild_button"), key=f"rebuild_{cache_id}"):
                        with st.spinner(_("rebuilding_cache")):
                            if rebuild_cache(cache_id):
                                st.success(_("rebuild_success"))
                            else:
                                st.error(_("rebuild_failed"))
                            st.experimental_rerun()
                
                # 删除按钮
                with col4:
                    if st.button(_("delete_button"), key=f"delete_{cache_id}"):
                        if delete_cache(cache_id):
                            st.success(_("delete_success"))
                            if cache_id in st.session_state.selected_caches:
                                st.session_state.selected_caches.remove(cache_id)
                            st.experimental_rerun()
                        else:
                            st.error(_("delete_failed"))
                
                st.markdown("---")
    else:
        st.info(_("no_cache_found"))


def main():
    """主应用程序入口"""
    
    # 会话状态初始化
    if 'model_download_progress' not in st.session_state:
        st.session_state.model_download_progress = 0.0
        
    if 'model_download_message' not in st.session_state:
        st.session_state.model_download_message = ""
        
    if 'is_model_ready' not in st.session_state:
        st.session_state.is_model_ready = False
        
    if 'selected_folders' not in st.session_state:
        st.session_state.selected_folders = []

    if 'show_file_explorer' not in st.session_state:
        st.session_state.show_file_explorer = False
        
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
        
    if 'last_model_type' not in st.session_state:
        st.session_state.last_model_type = ""
        
    if 'last_model_name' not in st.session_state:
        st.session_state.last_model_name = ""
        
    if 'last_force_rebuild' not in st.session_state:
        st.session_state.last_force_rebuild = False
        
    # 缓存管理相关状态
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "search"  # 默认显示搜索页面

    # 处理从历史记录重新执行搜索的标签页切换
    if st.session_state.get('switch_to_search_tab', False):
        st.session_state.selected_tab = "search"
        st.session_state.switch_to_search_tab = False
        # 如果有保存的模型类型，也设置它
        if 'repeat_search_model_type' in st.session_state:
            st.session_state.last_model_type = st.session_state.repeat_search_model_type
    
    if 'selected_caches' not in st.session_state:
        st.session_state.selected_caches = []
        
    if 'search_keyword' not in st.session_state:
        st.session_state.search_keyword = ""
        
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
        
    # 下载进度回调函数
    def update_download_progress(message: str, progress: float):
        st.session_state.model_download_message = message
        st.session_state.model_download_progress = progress
        if progress >= 1.0:
            st.session_state.is_model_ready = True
    
    # 侧边栏：应用设置
    with st.sidebar:
        st.title(f"{_('app_icon')} {_('app_title')}")
        st.markdown("---")
        
        # 语言选择
        st.subheader(_("language_settings"))
        lang_cols = st.columns(len(SUPPORTED_LANGUAGES))
        
        for i, (lang_code, lang_name) in enumerate(SUPPORTED_LANGUAGES.items()):
            with lang_cols[i]:
                # 如果是当前语言，显示为选中状态
                if st.button(
                    _("chinese" if lang_code == "zh" else "english"),
                    disabled=st.session_state.language == lang_code,
                    key=f"lang_{lang_code}"
                ):
                    change_language(lang_code)
                    st.rerun()
        
        st.markdown("---")
        
        # 标签选择
        tab_options = ["search", "search_history", "favorites", "tags", "cache_management"]
        selected_tab = st.radio(
            _("tab_select"),
            tab_options,
            format_func=lambda x: (
                _("search_tab") if x == "search" else
                (_("search_history_tab") if x == "search_history" else
                 (_("favorites_tab") if x == "favorites" else
                  (_("tags_tab") if x == "tags" else _("cache_tab"))))
            ),
            key="selected_tab",
            index=tab_options.index(st.session_state.selected_tab)
        )
        
        st.markdown("---")
        
        # 根据选择的标签显示不同的侧边栏内容
        if st.session_state.selected_tab == "search":
            # 模型选择
            st.subheader(_("model_settings"))
            
            # 模型类型选择
            model_type_options = [_("model_clip"), _("model_resnet"), _("model_custom")]
            model_type_selection = st.radio(
                _("select_model_type"),
                model_type_options,
                index=0,
                help=_("model_help")
            )
            
            # 映射UI选择到代码中的模型类型
            model_type_map = {
                _("model_clip"): "clip",
                _("model_resnet"): "resnet",
                _("model_custom"): "custom"
            }
            model_type = model_type_map.get(model_type_selection, "clip")
            
            # 自定义模型名称输入
            model_name = None
            if model_type == "custom":
                model_name = st.text_input(
                    _("model_custom"), 
                    placeholder=_("custom_model_placeholder"),
                    help=_("custom_model_help")
                )
                if not model_name:
                    st.warning(_("custom_model_required"))
            else:
                # 显示默认模型信息
                default_model = DEFAULT_MODEL_CONFIG.get(model_type, {}).get('name', _("unknown"))
                st.markdown(f"<div class='model-info'>{_('using_model', model_name=default_model)}</div>", unsafe_allow_html=True)
            
            # 检查模型或配置是否已更改，如果是，则重置搜索状态
            if (st.session_state.last_model_type != model_type or 
                st.session_state.last_model_name != (model_name or "")):
                st.session_state.search_triggered = False
                st.session_state.last_model_type = model_type
                st.session_state.last_model_name = model_name or ""
                
            # 显示模型下载/加载进度
            if st.session_state.model_download_message:
                st.markdown(f"### {_('model_loading')}")
                st.markdown(f"<div class='download-progress'>{st.session_state.model_download_message}</div>", unsafe_allow_html=True)
                if st.session_state.model_download_progress > 0:
                    st.progress(st.session_state.model_download_progress)
                    
            # 模型重置按钮
            if st.session_state.is_model_ready:
                if st.button(_("reset_model"), help=_("reset_model_help")):
                    st.cache_resource.clear()
                    st.session_state.is_model_ready = False
                    st.session_state.model_download_progress = 0.0
                    st.session_state.model_download_message = _("model_reset_message")
                    st.session_state.search_triggered = False  # 重置模型时同时重置搜索状态
                    st.rerun()
            
            # 搜索设置
            st.subheader(_("search_settings"))

            # 检查是否是自动执行搜索，如果是则使用保存的结果数量
            auto_execute = st.session_state.get('auto_execute_search', False)
            if auto_execute and 'repeat_search_max_results' in st.session_state:
                # 自动执行搜索时，使用保存的结果数量，但仍显示滑块（禁用状态）
                saved_max_results = st.session_state.repeat_search_max_results
                top_n = st.slider(
                    _("result_count"),
                    min_value=1,
                    max_value=50,
                    value=saved_max_results,
                    step=1,
                    help=_("result_count_help"),
                    disabled=True  # 自动执行时禁用滑块
                )
                st.info(f"🔄 使用历史记录中的结果数量: {saved_max_results}")
            else:
                # 正常情况下的滑块
                top_n = st.slider(
                    _("result_count"),
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    help=_("result_count_help")
                )
            
            # 路径输入
            st.subheader(_("image_library_path"))
            
            # 检查streamlit-folium是否可用
            if not STREAMLIT_FOLIUM_AVAILABLE:
                st.error(_("streamlit_folium_error"))
            else:
                # 显示/隐藏文件资源管理器的按钮
                if st.button(_("open_file_explorer"), help=_("file_explorer_help")):
                    st.session_state.show_file_explorer = True
                    st.rerun()

            # 显示已选择的文件夹列表
            if st.session_state.selected_folders:
                st.markdown(f"### {_('selected_folders')}")
                
                # 创建一个容器来显示文件夹列表
                with st.container():
                    for i, folder in enumerate(st.session_state.selected_folders):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.text(folder)
                        with col2:
                            if st.button(_("delete"), key=f"del_{i}"):
                                st.session_state.selected_folders.pop(i)
                                st.rerun()
            
                # 创建文本表示，用于后续处理
                folder_paths_text = "\n".join(st.session_state.selected_folders)
            else:
                folder_paths_text = ""
                st.info(_("select_folders"))
            
            # 添加清除所有文件夹按钮
            if st.session_state.selected_folders:
                if st.button(_("clear_folders")):
                    st.session_state.selected_folders = []
                    st.rerun()
                    
            # 手动输入路径（作为备选方案）
            with st.expander(_("manual_input")):
                manual_paths = st.text_area(
                    _("manual_input_label"),
                    placeholder=_("manual_input_placeholder"),
                    height=100
                )
                
                # 如果手动输入了路径，添加到选择列表
                if manual_paths and st.button(_("add_manual_paths")):
                    new_paths = [p.strip() for p in manual_paths.split("\n") if p.strip()]
                    for path in new_paths:
                        if path not in st.session_state.selected_folders:
                            st.session_state.selected_folders.append(path)
                    st.rerun()

            force_rebuild = st.checkbox(
                _("force_rebuild"),
                value=False,
                help=_("force_rebuild_help")
            )
            
            # 检查强制重建索引选项是否更改
            if st.session_state.last_force_rebuild != force_rebuild:
                st.session_state.search_triggered = False
                st.session_state.last_force_rebuild = force_rebuild

        elif st.session_state.selected_tab == "search_history":
            # 搜索历史侧边栏选项
            st.subheader(_("search_history"))

            # 历史记录筛选选项
            history_filter_type = st.selectbox(
                _("filter_by_model"),
                ["all", "clip", "resnet", "custom"],
                format_func=lambda x: _("all_searches") if x == "all" else x.upper(),
                key="history_model_filter"
            )

            # 日期筛选
            date_filter = st.date_input(
                _("filter_by_date"),
                value=None,
                key="history_date_filter"
            )

            # 仅显示收藏
            favorites_only = st.checkbox(
                _("favorite_searches"),
                value=False,
                key="history_favorites_only"
            )

            # 搜索历史记录
            history_search_term = st.text_input(
                _("search_in_history"),
                placeholder=_("search_history_placeholder"),
                key="history_search_term"
            )

            # 清除筛选按钮
            if st.button(_("clear_filters")):
                st.session_state.history_model_filter = "all"
                st.session_state.history_date_filter = None
                st.session_state.history_favorites_only = False
                st.session_state.history_search_term = ""
                st.rerun()

            # 历史记录管理按钮
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(_("export_history")):
                    st.session_state.show_export_dialog = True
            with col2:
                if st.button(_("import_history")):
                    st.session_state.show_import_dialog = True

            # 清空历史记录按钮
            if st.button(_("clear_all_history"), type="secondary"):
                st.session_state.show_clear_history_dialog = True

            # 清理孤儿图像按钮
            if st.button(f"🧹 {_('cleanup_orphaned_images')}", help=_("cleanup_orphaned_images_help")):
                st.session_state.show_cleanup_images_dialog = True

        elif st.session_state.selected_tab == "favorites":
            # 收藏管理侧边栏选项
            st.subheader(_("favorites_management"))

            # 收藏筛选选项
            favorites_sort_by = st.selectbox(
                _("favorites_sort_by"),
                ["newest", "oldest", "most_similar", "execution_time"],
                format_func=lambda x: {
                    "newest": _("sort_newest"),
                    "oldest": _("sort_oldest"),
                    "most_similar": _("sort_most_similar"),
                    "execution_time": _("sort_execution_time")
                }[x],
                key="favorites_sort_by"
            )

            # 模型筛选
            favorites_model_filter = st.selectbox(
                _("favorites_model_filter"),
                ["all", "clip", "resnet"],
                format_func=lambda x: _("all_models") if x == "all" else x.upper(),
                key="favorites_model_filter"
            )

            # 收藏管理操作
            st.markdown("---")
            if st.button(_("unfavorite_all"), type="secondary"):
                st.session_state.show_unfavorite_all_dialog = True

        elif st.session_state.selected_tab == "tags":
            # 标签管理侧边栏选项
            st.subheader(f"🏷️ {_('tags_management')}")

            # 显示标签统计概览
            try:
                history_manager = SearchHistoryManager()
                all_tags = history_manager.get_all_tags()

                if all_tags:
                    st.info(f"📊 {_('tag_info_sidebar', count=len(all_tags))}")

                    # 显示最常用的标签
                    top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:5]
                    st.markdown(f"**🔥 {_('most_used_tags')}：**")
                    for tag, count in top_tags:
                        st.markdown(f"• {tag} ({count})")
                else:
                    st.info(f"📝 {_('no_tags_yet')}")
            except Exception as e:
                st.error(f"获取标签信息失败: {e}")

        else:  # 缓存管理标签
            # 缓存管理选项
            st.subheader(_("cache_management_options"))

            # 缓存搜索
            search_keyword = st.text_input(_("search_cache_placeholder"), value=st.session_state.search_keyword, key="sidebar_cache_search")
            if search_keyword != st.session_state.search_keyword:
                st.session_state.search_keyword = search_keyword
                st.session_state.current_page = 0

            # 刷新按钮
            if st.button(_("refresh_cache_button")):
                st.rerun()
                
        # 硬件信息部分
        with st.expander(_("hardware_info")):
            # 检测GPU可用性
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                
                st.success(_("gpu_detected").format(count=gpu_count))
                st.markdown(_("gpu_name").format(name=gpu_name))
                st.markdown(_("cuda_version").format(version=torch.version.cuda))
            else:
                st.warning(_("no_gpu_warning"))
                st.markdown(_("device_cpu"))
        
    # 主区域内容：根据选择的标签显示不同内容
    if st.session_state.selected_tab == "search":
        # 搜索界面
        st.title(_("upload_title"))
        st.markdown(_("upload_description"))
        
        # 显示文件资源管理器
        if st.session_state.show_file_explorer:
            st.subheader(_("file_explorer"))
            st.caption(_("browse_folders"))
            
            selected_path = folium_file_explorer()
            
            if selected_path:
                if selected_path not in st.session_state.selected_folders:
                    st.session_state.selected_folders.append(selected_path)
                st.session_state.show_file_explorer = False
                st.rerun()
                
            # 添加返回按钮
            if st.button(_("return_to_main"), key="return_from_explorer"):
                st.session_state.show_file_explorer = False
                st.rerun()
        else:
            # 检查是否是自动执行搜索（从历史记录重新执行）
            auto_execute = st.session_state.get('auto_execute_search', False)
            repeat_image_path = st.session_state.get('repeat_search_image_path', '')

            # 上传图片（如果不是自动执行搜索）
            if not auto_execute:
                uploaded_file = st.file_uploader(_("upload_image"), type=["jpg", "jpeg", "png", "bmp", "webp"])
            else:
                uploaded_file = None  # 自动执行时不需要上传

            # 状态和结果容器
            status_container = st.empty()
            search_button_container = st.container()
            progress_bar = st.empty()
            results_container = st.container()

            # 处理搜索
            if uploaded_file or auto_execute:
                if auto_execute:
                    # 自动执行搜索：使用历史记录中的图像
                    query_image_path = repeat_image_path
                    st.info(_("using_historical_image"))
                else:
                    # 手动上传：处理上传的文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        query_image_path = tmp_file.name

                # 显示查询图像
                query_image = load_image(query_image_path)
                
                if query_image:
                    # 在上方显示查询图像
                    st.subheader(_("query_image"))
                    st.image(query_image, width=300)
                    
                    # 添加搜索按钮（如果不是自动执行）
                    if not auto_execute:
                        with search_button_container:
                            if st.session_state.selected_folders:
                                if st.button(_("start_search"), type="primary", help=_("search_button_help"), key="search_button"):
                                    st.session_state.search_triggered = True
                                    st.rerun()
                            else:
                                st.warning(_("folder_required"))
                    else:
                        # 自动执行搜索：直接触发搜索
                        st.session_state.search_triggered = True

                    # 只有在触发搜索后才执行搜索
                    if st.session_state.search_triggered and st.session_state.selected_folders:
                        # 验证自定义模型名称
                        if model_type == "custom" and not model_name:
                            status_container.error(_("custom_model_required"))
                        else:
                            # 使用会话中存储的文件夹路径
                            folder_paths = st.session_state.selected_folders
                            valid_paths, invalid_paths = validate_paths(folder_paths)
                            
                            if not valid_paths:
                                status_container.error(_("all_paths_invalid"))
                            else:
                                if invalid_paths:
                                    st.warning(_("invalid_paths") + "\n- ".join(invalid_paths))
                                    
                                # 搜索相似图像
                                with st.spinner(_("searching")):
                                    # 初始化进度条
                                    progress = 0.0
                                    progress_bar_obj = progress_bar.progress(progress)
                                    
                                    def update_progress(p: float):
                                        nonlocal progress
                                        progress = p
                                        progress_bar_obj.progress(progress)
                                        
                                    # 获取相似度搜索实例
                                    search = get_similarity_search(
                                        model_type=model_type,
                                        model_name=model_name,
                                        _progress_callback=update_download_progress
                                    )
                                    
                                    # 执行搜索
                                    start_time = time.time()
                                    try:
                                        # 检查是否是重新执行搜索（需要复用图像）
                                        reuse_image_path = st.session_state.get('repeat_search_stored_path', None)

                                        results = search.search_similar_images(
                                            query_image_path=query_image_path,
                                            folder_paths=valid_paths,
                                            top_n=top_n,
                                            force_rebuild_index=force_rebuild,
                                            progress_callback=update_progress,
                                            reuse_image_path=reuse_image_path
                                        )
                                        elapsed_time = time.time() - start_time
                                    except Exception as e:
                                        status_container.error(_("search_error").format(error=str(e)))
                                        results = []
                                        elapsed_time = time.time() - start_time
                                
                                # 清理临时文件（仅对手动上传的文件）
                                if not auto_execute:
                                    try:
                                        os.unlink(query_image_path)
                                    except Exception:
                                        pass

                                # 清理重新执行搜索的状态
                                if auto_execute:
                                    st.session_state.auto_execute_search = False
                                    if 'repeat_search_model_name' in st.session_state:
                                        del st.session_state.repeat_search_model_name
                                    if 'repeat_search_model_type' in st.session_state:
                                        del st.session_state.repeat_search_model_type
                                    if 'repeat_search_image_path' in st.session_state:
                                        del st.session_state.repeat_search_image_path
                                    if 'repeat_search_max_results' in st.session_state:
                                        del st.session_state.repeat_search_max_results
                                    if 'repeat_search_stored_path' in st.session_state:
                                        del st.session_state.repeat_search_stored_path
                                
                                # 显示结果
                                if results:
                                    status_container.success(_("search_success").format(count=len(results), time=elapsed_time))
                                    
                                    with results_container:
                                        st.subheader(_("search_results"))
                                        st.markdown(_("sorted_by"))
                                        
                                        # 创建结果表格列表
                                        result_data = []
                                        
                                        # 创建网格布局
                                        cols = st.columns(3)  # 每行3列
                                        
                                        for idx, result in enumerate(results):
                                            col_idx = idx % len(cols)
                                            similarity = result["similarity"]
                                            path = result["path"]
                                            
                                            # 准备表格数据
                                            result_data.append({
                                                _("table_index"): idx + 1,
                                                _("table_similarity"): f"{similarity:.4f}",
                                                _("table_filepath"): path
                                            })
                                            
                                            # 在网格中显示图像和信息
                                            with cols[col_idx]:
                                                try:
                                                    image = load_image(path)
                                                    if image:
                                                        st.image(
                                                            image,
                                                            caption=_("similarity").format(value=similarity),
                                                            use_container_width=True
                                                        )
                                                        # 使用可下载链接替换纯文本路径
                                                        st.markdown(_("path").format(path=get_file_download_link(path)), unsafe_allow_html=True)
                                                        st.write("---")
                                                except Exception as e:
                                                    st.error(f"无法加载图像 {path}: {e}")
                                                    
                                        # 显示表格数据
                                        st.subheader(_("results_table"))
                                        st.dataframe(pd.DataFrame(result_data), hide_index=True)
                                        
                                else:
                                    status_container.error(_("no_results"))
                                
                                # 添加重置搜索按钮
                                if st.button(_("reset_search"), key="reset_search"):
                                    st.session_state.search_triggered = False
                                    st.rerun()
                else:
                    status_container.error(_("image_load_error"))
            else:
                status_container.info(_("upload_prompt"))
    elif st.session_state.selected_tab == "search_history":
        # 搜索历史界面
        show_search_history()
    elif st.session_state.selected_tab == "favorites":
        # 收藏管理界面
        show_favorites_management()
    elif st.session_state.selected_tab == "tags":
        # 标签管理界面
        show_tags_management()
    else:
        # 缓存管理界面
        show_cache_management()


if __name__ == "__main__":
    main() 