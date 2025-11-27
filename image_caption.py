import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import time
import numpy as np

def put_text_with_newlines(img, text, pos, font_face, font_scale, color, thickness, line_type):
    """
    A helper function to put text with newlines on an image using OpenCV.
    """
    lines = text.split('\n')
    y0, dy = pos[1], 35  # Initial y position and line height
    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(img, line, (pos[0], y), font_face, font_scale, color, thickness, line_type)

def main():
    # --- 1. 模型加载 ---
    model_path = "./model"

    # 从本地加载
    processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    
    # 检测并设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"模型已加载到设备: {device}")

    # --- 2. 摄像头设置 ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        return

    # 设置处理帧率
    fps_to_process = 1  # 每秒处理1帧
    frame_interval = 1.0 / fps_to_process
    last_process_time = 0

    print("\n摄像头已启动，正在生成描述...")
    print("按 'q' 键退出程序。")
    print("-" * 30)

    # --- 3. 预热 (可选但推荐) ---
    # 第一次推理可能较慢（GPU初始化等），先跑一次预热
    print("正在进行模型预热...")
    ret, frame = cap.read()
    if ret:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        inputs = processor(pil_image, return_tensors="pt")
        # 关键：预热时也要将输入移动到正确的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model.generate(**inputs)
    print("预热完成。")

    # --- 4. 主循环 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法从摄像头读取帧，退出中...")
            break

        current_time = time.time()
        
        # --- 模型推理与结果绘制 ---
        if current_time - last_process_time >= frame_interval:
            last_process_time = current_time
            try:
                # 转换图像格式
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # 准备模型输入
                inputs = processor(pil_image, return_tensors="pt")
                
                # 【核心修改】将输入数据移动到与模型相同的设备上
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 生成描述
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_length=150,           # 生成序列的最大长度，默认是20，可以适当增加
                        num_beams=5,             # 使用 Beam Search，5个候选序列，能显著提升质量
                        early_stopping=False,    # 不提前停止，让模型生成到 max_length
                        repetition_penalty=1.5,  # 惩罚重复，值越大惩罚越重
                        length_penalty=2,     # 鼓励生成长文本，值>1.0时越长越好
                        num_return_sequences=1   # 只返回一个最好的序列
                    )
                
                caption = processor.decode(out[0], skip_special_tokens=True)
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                print(f"[{timestamp}] 描述: {caption}")

                # 在图像上显示描述
                # 将描述文字包装以适应屏幕宽度
                max_width = 60
                wrapped_caption = [caption[i:i+max_width] for i in range(0, len(caption), max_width)]
                display_text = '\n'.join(wrapped_caption)
                put_text_with_newlines(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"处理帧时发生错误: {e}")
                # 在图像上显示错误信息
                cv2.putText(frame, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 显示视频流
        cv2.imshow('Camera Feed - BLIP Description', frame)

        # 检查退出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户请求退出。")
            break

    # --- 5. 清理资源 ---
    cap.release()
    cv2.destroyAllWindows()
    print("程序已结束。")

if __name__ == "__main__":
    main()
