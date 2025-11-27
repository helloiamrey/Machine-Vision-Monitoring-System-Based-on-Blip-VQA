import torch
import cv2
from PIL import Image
from modelscope import BlipProcessor, BlipForQuestionAnswering
import numpy as np
import time

class VQAInterface:
    """VQA接口类，用于处理图片问答"""
    
    def __init__(self, model_path="./vqa"):
        """初始化VQA模型"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前使用的设备: {device}")
        self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float16
        ).to(device)
        print(f"[{time.strftime('%H:%M:%S')}] VQA模型加载成功至设备: {device}")
    
    def _preprocess_image(self, image):
        """预处理图片，支持多种输入格式"""
        if isinstance(image, str):
            # 文件路径
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # numpy数组（OpenCV格式）
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            # PIL Image，直接使用
            pass
        else:
            raise ValueError(f"[{time.strftime('%H:%M:%S')}] 不支持的图片格式: {type(image)}")
        return image
    
    def answer_question(self, image, question):
        """对图片回答问题"""
        try:
            pil_image = self._preprocess_image(image)
            inputs = self.processor(pil_image, question, return_tensors="pt").to("cuda", torch.float16)
            
            with torch.no_grad():
                out = self.model.generate(**inputs)
                answer = self.processor.decode(out[0], skip_special_tokens=True)
            
            return answer
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] VQA处理出错: {e}")
            return "处理失败"
    
    def batch_answer_questions(self, image, questions):
        """批量回答多个问题"""
        results = []
        for question in questions:
            answer = self.answer_question(image, question)
            results.append({'question': question, 'answer': answer})
        return results
