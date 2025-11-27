import cv2
import numpy as np
# 打开摄像头或视频文件
cap = cv2.VideoCapture(0) # 参数0表示第一个摄像头
# 初始化前一帧
prev_frame = None
while True:
   ret, frame = cap.read()
   if not ret:
       break
   # 转换为灰度图并进行高斯滤波
   gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
   if prev_frame is None:
       prev_frame = gray_frame
       continue
   # 计算帧差并进行二值化处理
   diff = cv2.absdiff(prev_frame, gray_frame)
   _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
   # 膨胀操作填充孔洞
   thresh = cv2.dilate(thresh, None, iterations=2)
   # 检测轮廓并绘制矩形框
   contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
       if cv2.contourArea(contour) < 1500: # 忽略小面积噪声
           continue
       x, y, w, h = cv2.boundingRect(contour)
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
   # 显示结果
   cv2.imshow("Frame", frame)
   cv2.imshow("Thresh", thresh)
   # 按 'q' 键退出
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()