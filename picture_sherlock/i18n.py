"""
国际化（i18n）模块：提供多语言支持功能
"""
import os
import json
import sys
from typing import Dict, Any, Optional
import streamlit as st

# 支持的语言列表
SUPPORTED_LANGUAGES = {
    "zh": "中文",
    "en": "English"
}

# 默认语言
DEFAULT_LANGUAGE = "zh"

# 翻译文件目录
TRANSLATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translations")

class I18n:
    """国际化处理类，负责加载和管理翻译"""
    
    def __init__(self):
        """初始化I18n类"""
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = DEFAULT_LANGUAGE
        
        # 加载所有支持的语言翻译
        for lang_code in SUPPORTED_LANGUAGES.keys():
            self._load_translation(lang_code)
    
    def _load_translation(self, lang_code: str) -> None:
        """
        加载指定语言的翻译文件
        
        Args:
            lang_code: 语言代码，例如"zh"或"en"
        """
        translation_path = os.path.join(TRANSLATIONS_DIR, f"{lang_code}.json")
        try:
            with open(translation_path, "r", encoding="utf-8") as f:
                self.translations[lang_code] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"警告: 无法加载语言文件 {translation_path}: {str(e)}")
            # 如果无法加载，使用空字典作为后备
            self.translations[lang_code] = {}
    
    def set_language(self, lang_code: str) -> None:
        """
        设置当前语言
        
        Args:
            lang_code: 语言代码，例如"zh"或"en"
        """
        if lang_code in SUPPORTED_LANGUAGES:
            self.current_language = lang_code
        else:
            print(f"警告: 不支持的语言代码 {lang_code}，使用默认语言 {DEFAULT_LANGUAGE}")
            self.current_language = DEFAULT_LANGUAGE
    
    def get_text(self, key: str, **format_args) -> str:
        """
        根据当前语言获取翻译文本
        
        Args:
            key: 翻译键，对应翻译文件中的键名
            **format_args: 格式化参数，用于替换翻译文本中的占位符
            
        Returns:
            str: 翻译后的文本，如果未找到翻译则返回键名
        """
        # 获取当前语言的翻译
        translation = self.translations.get(self.current_language, {})
        
        # 尝试获取翻译文本
        text = translation.get(key)
        
        # 如果未找到翻译，尝试使用默认语言
        if text is None and self.current_language != DEFAULT_LANGUAGE:
            default_translation = self.translations.get(DEFAULT_LANGUAGE, {})
            text = default_translation.get(key)
        
        # 如果仍未找到翻译，返回键名
        if text is None:
            return key
        
        # 如果有格式化参数，进行替换
        if format_args:
            try:
                return text.format(**format_args)
            except KeyError as e:
                print(f"警告: 格式化文本时出错，缺少参数 {e}")
                return text
        
        return text


# 创建单例实例
_i18n_instance: Optional[I18n] = None

def get_i18n() -> I18n:
    """
    获取I18n单例实例
    
    Returns:
        I18n: I18n实例
    """
    global _i18n_instance
    if _i18n_instance is None:
        _i18n_instance = I18n()
    return _i18n_instance

def _(key: str, **format_args) -> str:
    """
    翻译函数，用于获取当前语言的文本
    
    Args:
        key: 翻译键
        **format_args: 格式化参数
        
    Returns:
        str: 翻译后的文本
    """
    return get_i18n().get_text(key, **format_args)

def get_lang_from_args() -> Optional[str]:
    """
    从命令行参数中获取语言设置
    
    Returns:
        Optional[str]: 语言代码，如果未找到则返回None
    """
    # 打印所有命令行参数用于调试
    print(f"Debug - Command line args: {sys.argv}")
    
    # 检查参数中是否有--lang参数
    for arg in sys.argv:
        # 处理格式为--lang=xx的参数
        if arg.startswith("--lang="):
            lang_code = arg.split("=")[1].strip()
            if lang_code in SUPPORTED_LANGUAGES:
                print(f"Debug - Found language from args: {lang_code}")
                return lang_code
        # 处理格式为--lang xx的参数
        elif arg == "--lang" and len(sys.argv) > sys.argv.index(arg) + 1:
            next_arg = sys.argv[sys.argv.index(arg) + 1]
            if next_arg in SUPPORTED_LANGUAGES:
                print(f"Debug - Found language from args: {next_arg}")
                return next_arg
    
    return None

def init_language() -> None:
    """
    初始化语言设置，优先从命令行参数获取，其次从会话状态中读取
    """
    # 首先尝试从命令行参数获取语言设置
    lang_from_args = get_lang_from_args()
    
    # 如果命令行参数指定了语言，且会话状态未初始化，则使用命令行参数
    if lang_from_args and 'language' not in st.session_state:
        st.session_state.language = lang_from_args
    # 否则确保会话状态中有语言选项
    elif 'language' not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE
        
    # 设置当前语言
    get_i18n().set_language(st.session_state.language)

def change_language(lang_code: str) -> None:
    """
    更改当前语言
    
    Args:
        lang_code: 语言代码
    """
    st.session_state.language = lang_code
    get_i18n().set_language(lang_code) 