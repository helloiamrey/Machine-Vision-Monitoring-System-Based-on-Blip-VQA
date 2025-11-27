#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr  # 关键：导入formataddr函数
import cv2
import numpy as np
from io import BytesIO
import datetime
from config_loader import CONFIG

# 邮件配置（从附件中获取）
my_sender = CONFIG["email"]["sender"]  # 发信人邮箱
my_pass = CONFIG["email"]["password"]     # 发件人邮箱授权码
my_user = CONFIG["email"]["receiver"]  # 收件人邮箱

def send_frame_as_email(frame,mail_msg):
    """
    将摄像头捕获的图像帧作为邮件发送
    :param frame: numpy数组格式的图像帧
    """
    # 创建邮件对象
    msgRoot = MIMEMultipart('related')
    # 修正：使用formataddr正确格式化邮件头部
    msgRoot['From'] = formataddr([CONFIG["email"]["sender_name"], my_sender])
    msgRoot['To'] = formataddr([CONFIG["email"]["receiver_name"], my_user])
    msgRoot['Subject'] = Header("紧急警报：检测到异常活动", 'utf-8')
    
    # 创建HTML内容
    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)
    
    # 邮件正文（包含图片引用）
    #mail_msg = """
    #<p>监控系统检测到潜在的紧急情况，请立即查看！</p>
    #<p>现场图像：</p>
    #<p><img src="cid:alert_image"></p>
    #<p><small>此邮件由自动监控系统发送于 {timestamp}</small></p>
    #""".format(timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    msgAlternative.attach(MIMEText(mail_msg, 'html', 'utf-8'))
    
    # 将图像帧转换为邮件附件
    # 使用BytesIO避免保存临时文件
    is_success, im_buf_arr = cv2.imencode(".jpg", frame)
    byte_im = BytesIO(im_buf_arr.tobytes())
    
    # 创建MIMEImage对象
    msgImage = MIMEImage(byte_im.read())
    msgImage.add_header('Content-ID', '<alert_image>')
    msgRoot.attach(msgImage)
    
    # 发送邮件
    try:
        # 使用QQ邮箱的SMTP_SSL服务器
        server = smtplib.SMTP_SSL(CONFIG["email"]["smtp_server"], CONFIG["email"]["smtp_port"])
        server.login(my_sender, my_pass)
        server.sendmail(my_sender, [my_user], msgRoot.as_string())
        server.quit()
        print("警报邮件发送成功")
        return True
    except Exception as e:
        print(f"邮件发送失败: {str(e)}")
        return False

# 示例使用
if __name__ == "__main__":
    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    
    if not cap.isOpened():
        print("无法打开摄像头")
    else:
        # 捕获一帧图像
        ret, frame = cap.read()
        if ret:
            # 发送包含图像帧的警报邮件
            send_frame_as_email(frame,"test mail")
        else:
            print("无法捕获图像")
    
    # 释放资源
    cap.release()
