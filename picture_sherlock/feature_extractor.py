"""
图像特征提取模块：负责加载预训练模型并提取图像特征
"""
import os
import sys
import time
from typing import Tuple, Union, Any, Dict, Optional, Callable

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError

# CLIP模型相关
try:
    from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
# ResNet模型相关
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# 添加导入模块的路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 优先使用绝对导入
try:
    # 尝试绝对导入
    from picture_sherlock.utils import logger
    from picture_sherlock.i18n import _
except ImportError:
    # 尝试直接导入
    from utils import logger
    try:
        from i18n import _
    except ImportError:
        # 如果导入失败，提供一个简单的替代函数
        def _(text, **kwargs):
            return text


# 默认模型配置
DEFAULT_MODEL_CONFIG = {
    'clip': {
        'name': 'openai/clip-vit-base-patch32',
        'processor_cls': CLIPProcessor,
        'model_cls': CLIPModel,
        'model_attr': 'vision_model'  # CLIP模型中提取视觉编码器的属性名
    },
    'resnet': {
        'name': 'resnet50',
        'pretrained': True
    }
}


class FeatureExtractor:
    """
    图像特征提取器类：加载预训练模型并提取图像特征向量
    
    支持的模型：
    - CLIP (首选): 使用OpenAI的CLIP模型的图像编码器部分
    - ResNet50 (备选): 使用预训练的ResNet50模型
    - 自定义: 其他兼容的Hugging Face模型
    """
    
    def __init__(self, 
                model_type: str = 'clip', 
                model_name: Optional[str] = None,
                device: Optional[str] = None,
                download_progress_callback: Optional[Callable[[str, float], None]] = None):
        """
        初始化特征提取器
        
        Args:
            model_type: 模型类型，'clip'、'resnet'或'custom'
            model_name: 模型名称，默认为None则使用预设值
            device: 计算设备，'cuda'或'cpu'，如果为None则自动选择
            download_progress_callback: 下载进度回调函数，接收消息和进度值(0-1)
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.download_progress_callback = download_progress_callback
        
        # 确定计算设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"使用 {self.device} 进行特征提取")
        
        # 加载模型和预处理器
        self.model = None
        self.processor = None
        
        # 初始化模型
        self.load_model()
        
    def _report_progress(self, message: str, progress: float = 0.0) -> None:
        """
        报告进度
        
        Args:
            message: 进度消息
            progress: 进度值(0-1)
        """
        logger.info(message)
        if self.download_progress_callback:
            self.download_progress_callback(message, progress)
            
    def _check_dependencies(self) -> bool:
        """
        检查必要的依赖项是否已安装
        
        Returns:
            bool: 如果所有必要的依赖项已安装则返回True，否则返回False
        """
        missing_deps = []
        
        if self.model_type in ['clip', 'custom'] and not TRANSFORMERS_AVAILABLE:
            missing_deps.append("transformers")
            
        if self.model_type == 'resnet' and not TORCHVISION_AVAILABLE:
            missing_deps.append("torchvision")
            
        if missing_deps:
            error_msg = _("missing_dependencies", deps=', '.join(missing_deps))
            logger.error(error_msg)
            self._report_progress(error_msg, 0.0)
            return False
            
        return True
        
    def load_model(self) -> None:
        """
        根据指定的类型加载预训练模型和预处理器
        """
        start_time = time.time()
        
        # 检查依赖项
        if not self._check_dependencies():
            raise ImportError(_("missing_dependencies_error"))
            
        try:
            # 确定模型名称
            if not self.model_name and self.model_type in DEFAULT_MODEL_CONFIG:
                if self.model_type == 'clip':
                    self.model_name = DEFAULT_MODEL_CONFIG['clip']['name']
                elif self.model_type == 'resnet':
                    self.model_name = DEFAULT_MODEL_CONFIG['resnet']['name']
            
            # 加载模型
            if self.model_type == 'clip' and TRANSFORMERS_AVAILABLE:
                self._report_progress(_("loading_clip_model", model_name=self.model_name), 0.1)
                
                try:
                    # 使用CLIP模型
                    self.model = CLIPModel.from_pretrained(self.model_name).vision_model
                    self.processor = CLIPProcessor.from_pretrained(self.model_name)
                    
                    self._report_progress(_("clip_model_loaded"), 0.9)
                except Exception as e:
                    logger.error(f"加载CLIP模型失败: {e}")
                    raise ValueError(_("clip_model_load_failed", model_name=self.model_name, error=str(e)))
                    
            elif self.model_type == 'custom' and TRANSFORMERS_AVAILABLE:
                if not self.model_name:
                    raise ValueError(_("custom_model_required"))
                    
                self._report_progress(_("loading_custom_model", model_name=self.model_name), 0.1)
                
                try:
                    # 使用自定义Hugging Face模型
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    
                    self._report_progress(_("custom_model_loaded"), 0.9)
                except Exception as e:
                    logger.error(f"加载自定义模型失败: {e}")
                    raise ValueError(_("custom_model_load_failed", model_name=self.model_name, error=str(e)))
                    
            elif self.model_type == 'resnet' and TORCHVISION_AVAILABLE:
                self._report_progress(_("loading_resnet_model"), 0.1)
                
                try:
                    # 使用ResNet50模型
                    self.model = models.resnet50(pretrained=True)
                    # 移除最后的分类层，只保留特征提取部分
                    self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
                    
                    # 为ResNet50定义预处理转换
                    self.processor = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        ),
                    ])
                    
                    self._report_progress(_("resnet_model_loaded"), 0.9)
                except Exception as e:
                    logger.error(f"加载ResNet50模型失败: {e}")
                    raise ValueError(_("resnet_model_load_failed", error=str(e)))
                    
            else:
                raise ValueError(_("unsupported_model_type", model_type=self.model_type))
                
            # 将模型移动到指定设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
            elapsed_time = time.time() - start_time
            self._report_progress(_("model_load_complete", time=elapsed_time), 1.0)
            
        except Exception as e:
            error_msg = _("model_load_failed", error=str(e))
            logger.error(error_msg)
            self._report_progress(error_msg, 0.0)
            raise
            
    def preprocess_image(self, image_path: str) -> Any:
        """
        预处理图像，将其转换为模型所需的格式
        
        Args:
            image_path: 图像文件的路径
            
        Returns:
            预处理后的图像数据，格式取决于模型类型
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.model_type in ['clip', 'custom']:
                # CLIP/自定义模型预处理
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                return inputs
                
            elif self.model_type == 'resnet':
                # ResNet预处理
                tensor = self.processor(image).unsqueeze(0)  # 添加批次维度
                return tensor.to(self.device)
                
        except UnidentifiedImageError:
            logger.warning(f"无法识别图像格式: {image_path}")
            return None
        except Exception as e:
            logger.error(f"图像预处理失败: {image_path}, 错误: {e}")
            return None
            
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        从图像中提取特征向量
        
        Args:
            image_path: 图像文件的路径
            
        Returns:
            np.ndarray: 图像的特征向量，如果处理失败则返回None
        """
        try:
            with torch.no_grad():
                preprocessed = self.preprocess_image(image_path)
                
                if preprocessed is None:
                    return None
                    
                if self.model_type in ['clip', 'custom']:
                    # CLIP/自定义模型特征提取
                    outputs = self.model(**preprocessed)
                    
                    # 根据模型输出的不同字段获取特征，优先使用pooler_output
                    if hasattr(outputs, 'pooler_output'):
                        features = outputs.pooler_output.cpu().numpy()
                    elif hasattr(outputs, 'last_hidden_state'):
                        # 使用最后一个隐藏状态的平均值作为特征
                        features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    else:
                        # 尝试转换模型的直接输出
                        features = outputs[0].mean(dim=1).cpu().numpy() if isinstance(outputs, tuple) else outputs.cpu().numpy()
                    
                elif self.model_type == 'resnet':
                    # ResNet特征提取
                    features = self.model(preprocessed).squeeze().cpu().numpy()
                    if features.ndim == 1:
                        features = features.reshape(1, -1)
                
                # 标准化特征向量
                norm = np.linalg.norm(features, axis=1, keepdims=True)
                normalized_features = features / (norm + 1e-8)  # 添加小值避免除零
                
                return normalized_features
                
        except Exception as e:
            logger.error(f"特征提取失败: {image_path}, 错误: {e}")
            return None
            
    def __call__(self, image_path: str) -> Optional[np.ndarray]:
        """
        使对象可调用，提取图像特征
        
        Args:
            image_path: 图像文件的路径
            
        Returns:
            np.ndarray: 图像的特征向量，如果处理失败则返回None
        """
        return self.extract_features(image_path) 