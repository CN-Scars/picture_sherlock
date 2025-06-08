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
from typing import List, Dict, Tuple, Optional, Union, Any

# 添加导入模块的路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import streamlit as st
import pandas as pd
import torch  # 导入torch用于硬件检测
from PIL import Image

# 优先使用绝对导入
try:
    # 尝试绝对导入（当安装为包或添加到sys.path时）
    from picture_sherlock.utils import logger, validate_paths, CACHE_DIR
    from picture_sherlock.similarity_search import SimilaritySearch
    from picture_sherlock.feature_extractor import DEFAULT_MODEL_CONFIG
    from picture_sherlock.file_selector import folium_file_explorer
    from picture_sherlock.i18n import _, init_language, change_language, SUPPORTED_LANGUAGES
except ImportError:
    # 尝试直接导入（当在目录内运行时）
    from utils import logger, validate_paths, CACHE_DIR
    from similarity_search import SimilaritySearch
    from feature_extractor import DEFAULT_MODEL_CONFIG
    from file_selector import folium_file_explorer
    from i18n import _, init_language, change_language, SUPPORTED_LANGUAGES

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


def clear_index_cache() -> Tuple[bool, str]:
    """
    清除索引缓存文件
    
    Returns:
        Tuple[bool, str]: 是否成功删除和消息
    """
    try:
        # 获取所有索引缓存文件
        cache_files = glob.glob(os.path.join(CACHE_DIR, "image_index_*"))
        
        if not cache_files:
            return False, _("no_index_cache_found")
            
        # 删除所有索引缓存文件
        for file_path in cache_files:
            os.remove(file_path)
            
        return True, _("index_cache_deleted").format(count=len(cache_files))
        
    except Exception as e:
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
        
        return True, _("model_cache_deleted").format(size=size_mb)
        
    except Exception as e:
        return False, _("model_cache_delete_error").format(error=str(e))


def get_cache_info() -> Dict[str, Any]:
    """
    获取缓存信息
    
    Returns:
        Dict[str, Any]: 缓存信息字典
    """
    cache_info = {}
    
    # 索引缓存信息
    index_cache_files = glob.glob(os.path.join(CACHE_DIR, "image_index_*"))
    index_cache_size = sum(os.path.getsize(f) for f in index_cache_files if os.path.isfile(f))
    cache_info["index_cache_count"] = len(index_cache_files) // 2  # 每个索引有两个文件
    cache_info["index_cache_size"] = index_cache_size / (1024 * 1024)  # 转换为MB
    
    # Hugging Face模型缓存信息
    model_cache_size = 0
    if os.path.exists(HUGGINGFACE_CACHE_DIR):
        for dirpath, dirnames, filenames in os.walk(HUGGINGFACE_CACHE_DIR):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                model_cache_size += os.path.getsize(file_path)
    
    cache_info["model_cache_size"] = model_cache_size / (1024 * 1024)  # 转换为MB
    
    return cache_info


@st.cache_resource
def get_similarity_search(model_type: str, model_name: Optional[str] = None, _progress_callback=None) -> SimilaritySearch:
    """
    获取或创建相似度搜索实例（缓存以提高性能）
    
    Args:
        model_type: 模型类型 ('clip', 'resnet' 或 'custom')
        model_name: 模型名称，当model_type为'custom'时必须提供
        _progress_callback: 进度回调函数，添加下划线前缀表示不进行哈希
        
    Returns:
        SimilaritySearch: 相似度搜索实例
    """
    return SimilaritySearch(model_type, model_name, _progress_callback)


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    加载图像
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        Optional[Image.Image]: 加载的图像，如果加载失败则返回None
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(_("image_load_error").format(error=str(e)))
        return None


def get_file_download_link(file_path: str) -> str:
    """
    为文件路径创建可下载的链接
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 包含下载链接的HTML字符串
    """
    try:
        # 获取文件名
        file_name = os.path.basename(file_path)
        
        # 读取文件内容
        with open(file_path, "rb") as file:
            file_bytes = file.read()
        
        # 创建base64编码的下载链接
        b64 = base64.b64encode(file_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{file_path}</a>'
        return href
    except Exception as e:
        # 如果出现错误，返回原始路径
        return file_path


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
            
        # 添加缓存管理部分
        st.subheader(_("cache_management"))
        
        # 获取缓存信息
        cache_info = get_cache_info()
        
        # 显示缓存信息
        st.markdown("<div class='cache-info'>", unsafe_allow_html=True)
        st.markdown(_("index_cache").format(count=cache_info['index_cache_count'], size=cache_info['index_cache_size']))
        st.markdown(_("model_cache").format(size=cache_info['model_cache_size']))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 添加清除索引缓存按钮
        if st.button(_("clear_index_cache"), help=_("clear_index_help"), key="clear_index_cache"):
            success, message = clear_index_cache()
            if success:
                st.success(message)
            else:
                st.error(message)
            st.rerun()
            
        # 添加清除模型缓存按钮
        with st.expander(_("advanced_options")):
            st.warning(_("clear_model_warning"))
            
            if st.button(_("clear_model_cache"), help=_("clear_model_help"), key="clear_model_cache"):
                success, message = clear_model_cache()
                if success:
                    st.success(message)
                else:
                    st.error(message)
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
        
    # 主区域：上传图片和搜索
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
            
    # 显示关于信息已移至页面配置的about参数中


if __name__ == "__main__":
    main() 