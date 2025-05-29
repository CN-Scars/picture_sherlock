"""
文件选择器模块：提供文件夹选择功能
支持基于Tkinter的系统本地对话框和基于streamlit-folium的文件资源管理器
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog
from typing import Optional, List, Callable, Dict, Any
import streamlit as st
import os.path as osp
import math

# 尝试导入i18n模块
try:
    # 尝试绝对导入
    from picture_sherlock.i18n import _
except ImportError:
    # 尝试相对导入
    try:
        from i18n import _
    except ImportError:
        # 如果导入失败，提供一个简单的替代函数
        def _(text, **kwargs):
            return text

# 检测streamlit-folium是否可用
try:
    import folium
    from streamlit_folium import st_folium
    STREAMLIT_FOLIUM_AVAILABLE = True
except ImportError:
    STREAMLIT_FOLIUM_AVAILABLE = False


def select_folder(initial_dir: str = None) -> Optional[str]:
    """
    打开系统文件对话框选择文件夹
    
    Args:
        initial_dir: 初始目录，如果为None则使用当前目录
        
    Returns:
        str: 选择的文件夹路径，如果用户取消则返回None
    """
    if initial_dir is None:
        initial_dir = os.path.expanduser("~")
        
    # 确保初始目录存在
    if not os.path.exists(initial_dir):
        initial_dir = os.path.expanduser("~")
    
    # 创建一个隐藏的Tkinter根窗口
    root = tk.Tk()
    root.withdraw()
    
    # 防止Tkinter窗口闪烁
    root.attributes("-topmost", True)
    
    # 打开文件夹选择对话框
    folder_path = filedialog.askdirectory(
        parent=root,
        initialdir=initial_dir,
        title=_("file_select_folder")
    )
    
    # 销毁Tkinter窗口
    root.destroy()
    
    # 返回选择的文件夹路径（如果用户取消则为空字符串）
    if folder_path:
        return os.path.normpath(folder_path)
    return None


def select_folders(initial_dir: str = None) -> List[str]:
    """
    连续选择多个文件夹（用户可以多次选择，直到点击完成）
    
    Args:
        initial_dir: 初始目录，如果为None则使用当前目录
        
    Returns:
        List[str]: 选择的文件夹路径列表
    """
    selected_folders = []
    current_dir = initial_dir
    
    while True:
        folder = select_folder(current_dir)
        if not folder:
            break
            
        selected_folders.append(folder)
        current_dir = os.path.dirname(folder)  # 下次从上一次选择的父目录开始
    
    return selected_folders


def select_folder_in_thread(initial_dir: str = None, callback: Callable[[str], None] = None) -> None:
    """
    在后台线程中打开文件夹选择对话框，避免阻塞Streamlit应用
    
    Args:
        initial_dir: 初始目录
        callback: 回调函数，接收选择的文件夹路径作为参数
    """
    def _select_folder_thread():
        folder = select_folder(initial_dir)
        if callback and folder:
            callback(folder)
    
    # 在后台线程中运行
    thread = threading.Thread(target=_select_folder_thread)
    thread.daemon = True
    thread.start()
    
    return thread


def folium_file_explorer(start_dir: str = None) -> Optional[str]:
    """
    基于streamlit-folium的文件资源管理器
    
    Args:
        start_dir: 起始目录，如果为None则使用当前目录
        
    Returns:
        str: 选择的文件夹路径，如果用户取消则返回None
    """
    if not STREAMLIT_FOLIUM_AVAILABLE:
        st.error(_("streamlit_folium_error"))
        return None
        
    if start_dir is None:
        start_dir = os.path.expanduser("~")  # 默认为用户主目录
    
    # 确保起始目录存在
    if not os.path.exists(start_dir):
        start_dir = os.path.expanduser("~")
    
    # 当前路径状态
    if 'current_path' not in st.session_state:
        st.session_state.current_path = start_dir
    
    # 显示路径和返回上一级按钮
    col_path, col_back = st.columns([3, 1])
    
    with col_path:
        st.text_input(_("file_current_path"), st.session_state.current_path, disabled=True)
    
    with col_back:
        if st.button(_("file_parent_dir"), use_container_width=True):
            parent_dir = os.path.dirname(st.session_state.current_path)
            if os.path.exists(parent_dir):
                st.session_state.current_path = parent_dir
                st.rerun()
    
    # 尝试读取当前目录内容
    try:
        items = []
        for item in os.listdir(st.session_state.current_path):
            full_path = os.path.join(st.session_state.current_path, item)
            is_dir = os.path.isdir(full_path)
            
            # 只展示目录
            if is_dir:
                items.append({
                    'name': item,
                    'path': full_path,
                    'is_dir': is_dir
                })
        
        # 按名称排序，目录优先
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
        
        if not items:
            st.info(_("file_no_subfolders"))
        else:
            st.markdown(f"### {_('file_folder_list')}")
            
            # 定义每行显示的文件夹数量
            folders_per_row = 4
            
            # 计算需要多少行来显示所有文件夹
            total_rows = math.ceil(len(items) / folders_per_row)
            
            # 将文件夹按行组织
            for row_idx in range(total_rows):
                # 创建当前行的列
                cols = st.columns(folders_per_row)
                
                # 计算当前行的文件夹索引范围
                start_idx = row_idx * folders_per_row
                end_idx = min(start_idx + folders_per_row, len(items))
                
                # 在每列中显示一个文件夹
                for col_idx, item_idx in enumerate(range(start_idx, end_idx)):
                    item = items[item_idx]
                    with cols[col_idx]:
                        # 使用带有图标的按钮显示文件夹
                        if st.button(f"📁 {item['name']}", key=item['path'], use_container_width=True):
                            st.session_state.current_path = item['path']
                            st.rerun()
        
        # 选择当前目录按钮
        selected_path = None
        if st.button(_("file_select_current"), type="primary"):
            selected_path = st.session_state.current_path
            
        return selected_path
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None 