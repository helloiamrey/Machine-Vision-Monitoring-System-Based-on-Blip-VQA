import cv2
import numpy as np
from vqa_interface import VQAInterface
from image_caption_interface import ImageCaptionInterface
import subprocess
import time
from send_email_v2 import send_frame_as_email
import datetime
from config_loader import CONFIG
import os

def save_frame_to_shots(frame):
    """保存帧到/shots目录"""
    # 创建shots目录（如果不存在）
    shots_dir = CONFIG["emergency"]["shots_path"]
    os.makedirs(shots_dir, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"shot_{timestamp}.jpg"
    filepath = os.path.join(shots_dir, filename)
    
    # 保存图片为JPEG格式
    success = cv2.imwrite(filepath, frame)
    if success:
        print(f"[{time.strftime('%H:%M:%S')}] 图片已保存到: {filepath}")
        return filepath
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 图片保存失败")
        return None


def main(frame=None):
    # 初始化VQA接口
    vqa = VQAInterface(model_path=CONFIG["models"]["vqa_path"])
    caption_interface = ImageCaptionInterface(CONFIG["models"]["image_caption_path"])


    # 处理摄像头图片
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    ret=1#模拟成功读取图片
    if ret:
        # 图片保存
        save_frame_to_shots(frame)
        # vqa问答
        questions = CONFIG["emergency"]["questions"]
        results = vqa.batch_answer_questions(frame, questions)
        for result in results:
            print(f"Q: {result['question']} -> A: {result['answer']}")
        if_emergency = all(result['answer'].lower() == 'yes' for result in results)
        if if_emergency:
            print(f"[{time.strftime('%H:%M:%S')}] 紧急情况检测到！")
            # 进一步询问
            try:
                cap.release()
                subprocess.run(["python", "emergency.py"], check=True)
                print(f"[{time.strftime('%H:%M:%S')}] 脚本执行完成")
                return True
            except subprocess.CalledProcessError as e:
                print(f"[{time.strftime('%H:%M:%S')}] 脚本执行错误: {e}")
                return False
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 脚本执行异常: {e}")
                return False
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 未检测到紧急情况。")


        questions2 = CONFIG["emergency"]["suspicious_questions"]
        results = vqa.batch_answer_questions(frame, questions2)
        for result in results:
            print(f"Q: {result['question']} -> A: {result['answer']}")
        if_suspicious = all(result['answer'].lower() == 'yes' for result in results)
        if if_suspicious:
            print(f"[{time.strftime('%H:%M:%S')}] 可疑人员检测到！")
            msg="""
            <p>监控系统检测到可疑人员.</p>
            <p>现场图像：</p>
            <p><img src="cid:alert_image"></p>
            <p><small>此邮件由自动监控系统发送于 {timestamp}</small></p>
            """.format(timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            send_frame_as_email(frame,msg)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 未检测到可疑人员。")

        



        # 图像描述
        single_caption = caption_interface.generate_caption(frame)
        print(f"摄像头图片描述: {single_caption}")
    cap.release()



#if __name__ == "__main__":
main()
