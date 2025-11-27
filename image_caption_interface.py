import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
import time

class ImageCaptionInterface:
    """图像描述生成接口类，支持摄像头图片处理"""
    
    def __init__(self, model_path="./model"):
        """初始化图像描述模型"""
        try:
            # 加载处理器和模型
            self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
            
            # 检测并设置设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            print(f"[{time.strftime('%H:%M:%S')}] 图像描述模型已加载到设备: {self.device}")
            
            # 预热模型
            self._warmup()
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 模型加载失败: {e}")
            raise
    
    def _warmup(self):
        """模型预热"""
        try:
            # 创建一个虚拟图像进行预热（模拟摄像头帧）
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.generate_caption(dummy_frame)
            print(f"[{time.strftime('%H:%M:%S')}] 模型预热完成")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 预热失败，但不影响使用: {e}")
    
    def _preprocess_image(self, image):
        """预处理图片，支持多种输入格式"""
        if isinstance(image, str):
            # 文件路径
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # numpy数组（OpenCV格式/摄像头帧）
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            # PIL Image，直接使用
            pass
        else:
            raise ValueError(f"[{time.strftime('%H:%M:%S')}] 不支持的图片格式: {type(image)}")
        return image
    
    def generate_caption(self, image, max_length=150, num_beams=5):
        """生成图像描述"""
        try:
            # 预处理图像
            pil_image = self._preprocess_image(image)
            
            # 准备输入
            inputs = self.processor(pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成描述
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=False,
                    repetition_penalty=1.5,
                    length_penalty=2,
                    num_return_sequences=1
                )
            
            # 解码结果
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 生成描述时出错: {e}")
            return "生成失败"
    
    def generate_captions_for_frame(self, frame, num_captions=3, **kwargs):
        """
        为单个摄像头帧生成多个描述（仿VQA的batch_answer_questions）
        
        Args:
            frame: 摄像头帧（numpy数组）
            num_captions: 要生成的描述数量
            **kwargs: generate_caption的其他参数
            
        Returns:
            list: 描述列表
        """
        results = []
        for i in range(num_captions):
            # 每次生成时稍微调整参数以获得不同的描述
            caption = self.generate_caption(
                frame, 
                num_beams=kwargs.get('num_beams', 5) + i,  # 改变beam数量
                max_length=kwargs.get('max_length', 150),
                repetition_penalty=1.5 + i * 0.1  # 调整重复惩罚
            )
            results.append({
                'caption_id': i + 1,
                'caption': caption
            })
        return results
    
    def batch_generate_captions(self, frames, **kwargs):
        """
        批量生成图像描述（完全仿VQA的batch_answer_questions）
        
        Args:
            frames: 图像列表（可以是文件路径、numpy数组、PIL Image的混合）
            **kwargs: generate_caption的其他参数
            
        Returns:
            list: 描述列表
        """
        results = []
        for i, frame in enumerate(frames):
            caption = self.generate_caption(frame, **kwargs)
            results.append({
                'frame_index': i,
                'caption': caption
            })
        return results
