import cv2
import numpy as np
from vqa_interface import VQAInterface
from send_email_v2 import send_frame_as_email
import pygame
import time
import datetime
from config_loader import CONFIG

def main():
    # 初始化
    vqa = VQAInterface(model_path=CONFIG["models"]["vqa_path"])
    
    # 处理一帧
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"[{time.strftime('%H:%M:%S')}] 错误：无法打开摄像头。")
        return

    ret, frame = cap.read()
    if not ret:
        print(f"[{time.strftime('%H:%M:%S')}] 错误：无法从摄像头读取帧。")
        cap.release()
        return

    # 为了让 cv2.waitKey() 能正常工作，需要显示一个窗口
    cv2.imshow('Emergency Monitor', frame)
    cv2.resizeWindow("Emergency Monitor", 640, 480)

    # VQA问答
    questions = CONFIG["emergency"]["questions"]
    results = vqa.batch_answer_questions(frame, questions)
    print(f"[{time.strftime('%H:%M:%S')}] VQA 问题及回答:")
    for result in results:
        print(f"Q: {result['question']} -> A: {result['answer']}")
    # 判断是否为紧急情况：所有问题都回答 'yes'
    if_emergency = all(result['answer'].lower() == 'yes' for result in results)
    


    #if_emergency = True  # 测试用，强制为紧急情况

    
    if if_emergency:
        print(f"[{time.strftime('%H:%M:%S')}] 紧急情况检测到！\n系统将在30秒内等待响应，否则将发送警报邮件。")
        print(f"[{time.strftime('%H:%M:%S')}] 请按任意键取消警报。")
        
        # --- 1. 启动声音警报 ---
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(CONFIG["emergency"]["alert_sound"])  # 确保 alert.mp3 文件存在
            pygame.mixer.music.play(-1)  # -1 表示循环播放
            print("声音警报已启动...")
        except pygame.error as e:
            print(f"[{time.strftime('%H:%M:%S')}] 无法播放声音警报: {e}")
            print(f"[{time.strftime('%H:%M:%S')}] 请确保 '{CONFIG["emergency"]["alert_sound"]}' 文件存在于脚本目录下。")

        # --- 2. 等待30秒或用户按键 ---
        wait_time_seconds = 30
        start_time = time.time()
        key_pressed = False

        while time.time() - start_time < wait_time_seconds:
            # cv2.waitKey(1) 会等待1毫秒，并检查是否有按键
            # 如果有按键，则返回按键的ASCII码；否则返回-1
            # & 0xFF 是为了兼容64位系统
            if cv2.waitKey(1) & 0xFF != 255:
                key_pressed = True
                break

        # --- 3. 根据结果执行后续操作 ---
        # 停止声音警报
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass # 如果pygame未初始化，忽略错误

        if key_pressed:
            print(f"[{time.strftime('%H:%M:%S')}] 用户已响应，警报已取消。")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 30秒内无响应，正在发送邮件警报...")
            mail_msg = """
            <p>监控系统检测到潜在的紧急情况，请立即查看！</p>
            <p>现场图像：</p>
            <p><img src="cid:alert_image"></p>
            <p><small>此邮件由自动监控系统发送于 {timestamp}</small></p>
            """.format(timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            send_frame_as_email(frame,mail_msg)
            #try:
            #    pygame.mixer.init()
            #    pygame.mixer.music.load(CONFIG["emergency"]["succeed_sound"])  # 确保 succeedsending.mp3 文件存在
            #    pygame.mixer.music.play()  
            #    while pygame.mixer.music.get_busy():
            #        time.sleep(0.1)
            #except pygame.error as e:
            #    print(f"[{time.strftime('%H:%M:%S')}] 无法播放声音警报: {e}")
            #    print(f"[{time.strftime('%H:%M:%S')}] 请确保 '{CONFIG["emergency"]["succeed_sound"]}' 文件存在于脚本目录下。")
            

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"[{time.strftime('%H:%M:%S')}] 程序结束。")



#if __name__ == "__main__":
main()
