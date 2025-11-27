import torch
import cv2
import requests
from PIL import Image
from modelscope import BlipProcessor, BlipForQuestionAnswering
import time

class RealTimeVQA:
    def __init__(self, model_path="./vqa", frame_interval=30):
        """
        初始化实时VQA系统
        :param model_path: 本地模型路径
        :param frame_interval: 处理间隔帧数（默认每30帧处理一次）
        """
        # 加载模型和处理器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前使用的设备: {device}")
        self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_path, 
            local_files_only=True, 
            dtype=torch.float16
        ).to(device)
        
        self.frame_interval = frame_interval
        self.frame_count = 0
        self.last_process_time = 0
        
        # 默认问题（可修改）
        self.question = "is the light bright enough?"

        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)  # 使用默认摄像头
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
            
        print("实时VQA系统已启动，按ESC退出")
        print(f"当前问题: {self.question}")
        print(f"处理间隔: 每{frame_interval}帧处理一次")

    def process_frame(self, frame):
        """处理单帧图像"""
        try:
            # 转换BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # 预处理并推理
            inputs = self.processor(pil_image, self.question, return_tensors="pt").to("cuda", torch.float16)
            
            with torch.no_grad():
                out = self.model.generate(**inputs)
                answer = self.processor.decode(out[0], skip_special_tokens=True)
            
            return answer
        except Exception as e:
            print(f"处理帧时出错: {e}")
            return "处理失败"

    def add_text_to_frame(self, frame, text, position=(50, 50)):
        """在帧上添加文字"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0)  # 绿色
        thickness = 2
        
        # 添加背景框
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, 
                     (position[0]-10, position[1]-text_height-10), 
                     (position[0]+text_width+10, position[1]+10), 
                     (0, 0, 0), -1)
        
        # 添加文字
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        
        return frame

    def run(self):
        """主运行循环"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                self.frame_count += 1
                current_time = time.time()
                
                # 检查是否需要处理当前帧
                if self.frame_count % self.frame_interval == 0:
                    print(f"处理第 {self.frame_count} 帧...")
                    
                    # 处理帧
                    answer = self.process_frame(frame)
                    self.last_answer = answer
                    self.last_process_time = current_time
                    
                    print(f"问题: {self.question}")
                    print(f"回答: {answer}")
                
                # 在帧上显示信息
                info_text = f"Frame: {self.frame_count} | Interval: {self.frame_interval}"
                frame = self.add_text_to_frame(frame, info_text, (10, 30))
                
                # 显示问题
                frame = self.add_text_to_frame(frame, f"Q: {self.question}", (10, 70))
                
                # 显示最新回答（如果有）
                if hasattr(self, 'last_answer'):
                    frame = self.add_text_to_frame(frame, f"A: {self.last_answer}", (10, 110))
                
                # 显示处理时间
                if self.last_process_time > 0:
                    time_since_process = current_time - self.last_process_time
                    frame = self.add_text_to_frame(frame, f"Last handled: {time_since_process:.1f}s before.", (10, 150))
                
                # 显示帧
                cv2.imshow('Real-time VQA', frame)
                
                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break
                elif key == ord('q'):  # Q键
                    # 可以在这里添加交互式问题输入
                    new_question = input("输入新问题（留空保持原问题）: ")
                    if new_question.strip():
                        self.question = new_question
                        print(f"问题已更新为: {self.question}")
                        
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("资源已释放")

if __name__ == "__main__":
    # 创建并运行实时VQA系统
    vqa_system = RealTimeVQA(
        model_path="./vqa",
        frame_interval=30  # 每30帧处理一次（约1秒，假设30fps）
    )
    
    vqa_system.run()
