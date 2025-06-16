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
except ImportError:
    # 尝试直接导入（当在目录内运行时）
    from utils import logger, validate_paths, CACHE_DIR
    from similarity_search import SimilaritySearch
    from feature_extractor import DEFAULT_MODEL_CONFIG
    from file_selector import folium_file_explorer
    from i18n import _, init_language, change_language, SUPPORTED_LANGUAGES
    from cache_manager import CacheManager

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
        tab_options = ["search", "cache_management"]
        selected_tab = st.radio(
            _("tab_select"),
            tab_options,
            format_func=lambda x: _("search_tab") if x == "search" else _("cache_tab"),
            key="selected_tab"
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
            # 上传图片
            uploaded_file = st.file_uploader(_("upload_image"), type=["jpg", "jpeg", "png", "bmp", "webp"])
            
            # 状态和结果容器
            status_container = st.empty()
            search_button_container = st.container()
            progress_bar = st.empty()
            results_container = st.container()
            
            # 处理搜索
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    query_image_path = tmp_file.name
                    
                # 显示查询图像
                query_image = load_image(query_image_path)
                
                if query_image:
                    # 在上方显示查询图像
                    st.subheader(_("query_image"))
                    st.image(query_image, width=300)
                    
                    # 添加搜索按钮
                    with search_button_container:
                        if st.session_state.selected_folders:
                            if st.button(_("start_search"), type="primary", help=_("search_button_help"), key="search_button"):
                                st.session_state.search_triggered = True
                                st.rerun()
                        else:
                            st.warning(_("folder_required"))
                    
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
                                        results = search.search_similar_images(
                                            query_image_path=query_image_path,
                                            folder_paths=valid_paths,
                                            top_n=top_n,
                                            force_rebuild_index=force_rebuild,
                                            progress_callback=update_progress
                                        )
                                        elapsed_time = time.time() - start_time
                                    except Exception as e:
                                        status_container.error(_("search_error").format(error=str(e)))
                                        results = []
                                        elapsed_time = time.time() - start_time
                                
                                # 清理临时文件
                                try:
                                    os.unlink(query_image_path)
                                except Exception:
                                    pass
                                
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
    else:
        # 缓存管理界面
        show_cache_management()


if __name__ == "__main__":
    main() 