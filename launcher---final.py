import requests
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
from datetime import datetime
import time
import cv2
import numpy as np
import threading
import queue
import pygame
import os
from vqa_interface import VQAInterface
from image_caption_interface import ImageCaptionInterface
from send_email_v2 import send_frame_as_email
from config_loader import CONFIG

# 全局模型实例
vqa_model = None
caption_model = None

def initialize_models():
    """一次性加载所有模型"""
    global vqa_model, caption_model
    if vqa_model is None:
        print(f"[{time.strftime('%H:%M:%S')}] 正在加载VQA模型...")
        vqa_model = VQAInterface(model_path=CONFIG["models"]["vqa_path"])
    if caption_model is None:
        print(f"[{time.strftime('%H:%M:%S')}] 正在加载图像描述模型...")
        caption_model = ImageCaptionInterface(CONFIG["models"]["image_caption_path"])

class MotionDetector:
    """运动检测类，集成VQA和图像处理功能"""
    
    def __init__(self, motion_threshold=1500, min_contour_area=500, 
                 motion_duration_threshold=0.5, sleep_timeout=10.0, 
                 emergency_threshold=1000, emergency_cooldown=30.0):
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.motion_duration_threshold = motion_duration_threshold
        self.sleep_timeout = sleep_timeout
        self.emergency_threshold = emergency_threshold
        self.emergency_cooldown = emergency_cooldown
        
        # 初始化前一帧
        self.prev_frame = None
        self.prev_sleep_frame = None
        
        # 运动状态跟踪
        self.motion_start_time = None
        self.is_motion_detected = False
        self.last_motion_time = None
        self.last_process_time = 0  # 最后一次处理时间
        self.process_interval = 5.0  # 处理间隔（秒）
        
        # 休眠/唤醒状态
        self.is_sleeping = True
        self.wake_time = None
        
        # 统计信息
        self.frame_count = 0
        self.sleep_frame_count = 0
        self.motion_frames = 0
        self.process_count = 0
        self.wake_count = 0
        self.emergency_count = 0
        
        # 休眠模式参数
        self.sleep_frame_skip = 5
        self.sleep_frame_counter = 0
        self.sleep_motion_threshold = 2000
        
        # 处理状态
        self.process_running = False
        self.pending_process = False
        
        # 紧急事件状态
        self.emergency_running = False
        self.last_emergency_time = 0
        
        # 初始化稳定标志
        self.initialization_frames = 0
        self.initialization_threshold = 10
    
    def _preprocess_frame(self, frame):
        """预处理帧"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray
    
    def _detect_motion(self, frame, is_sleep_mode=False):
        """检测运动"""
        gray_frame = self._preprocess_frame(frame)
        
        prev_frame = self.prev_sleep_frame if is_sleep_mode else self.prev_frame
        
        if prev_frame is None:
            if is_sleep_mode:
                self.prev_sleep_frame = gray_frame
            else:
                self.prev_frame = gray_frame
            return False, [], None, 0
        
        diff = cv2.absdiff(prev_frame, gray_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_contours = []
        total_motion_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = self.min_contour_area // 2 if is_sleep_mode else self.min_contour_area
            if area > min_area:
                significant_contours.append(contour)
                total_motion_area += area
        
        if is_sleep_mode:
            self.prev_sleep_frame = gray_frame
        else:
            self.prev_frame = gray_frame
        
        threshold = self.sleep_motion_threshold if is_sleep_mode else self.motion_threshold
        has_motion = total_motion_area > threshold
        
        return has_motion, significant_contours, thresh, total_motion_area
    
    def _should_process(self):
        """判断是否应该处理帧"""
        current_time = time.time()
        return (current_time - self.last_process_time) >= self.process_interval
    
    def _should_sleep(self):
        """判断是否应该进入休眠"""
        if self.is_sleeping:
            return False
        
        if self.last_motion_time is None:
            return False
        
        current_time = time.time()
        return (current_time - self.last_motion_time) >= self.sleep_timeout
    
    def save_frame_to_shots(self, frame):
        """保存帧到/shots目录"""
        shots_dir = CONFIG["emergency"]["shots_path"]
        os.makedirs(shots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shot_{timestamp}.jpg"
        filepath = os.path.join(shots_dir, filename)
        
        success = cv2.imwrite(filepath, frame)
        if success:
            print(f"[{time.strftime('%H:%M:%S')}] 图片已保存到: {filepath}")
            return filepath
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 图片保存失败")
            return None
    
    def process_frame_with_models(self, frame):
        """使用加载的模型处理帧"""
        if self.process_running:
            return
        
        self.process_running = True
        try:
            # 保存帧
            self.save_frame_to_shots(frame)
            # 图像描述
            if caption_model:
                single_caption = caption_model.generate_caption(frame)
                print(f"摄像头图片描述: {single_caption}")
            # VQA问答
            questions = CONFIG["emergency"]["questions"]
            results = vqa_model.batch_answer_questions(frame, questions)
            print(f"[{time.strftime('%H:%M:%S')}] VQA 问题及回答:")
            for result in results:
                print(f"Q: {result['question']} -> A: {result['answer']}")
            
            # 判断紧急情况
            if_emergency = all(result['answer'].lower() == 'yes' for result in results)
            
            if if_emergency:
                print(f"[{time.strftime('%H:%M:%S')}] 紧急情况检测到！")
                self.emergency_count += 1
                threading.Thread(target=self.emergency_process, args=(frame, results), daemon=True).start()
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 未检测到紧急情况。")
                
                # 可疑人员检测
                questions2 = CONFIG["emergency"]["suspicious_questions"]
                results2 = vqa_model.batch_answer_questions(frame, questions2)
                for result in results2:
                    print(f"Q: {result['question']} -> A: {result['answer']}")
                
                if_suspicious = all(result['answer'].lower() == 'yes' for result in results2)
                if if_suspicious:
                    print(f"[{time.strftime('%H:%M:%S')}] 可疑人员检测到！")
                    msg = f"""
                    <p>监控系统检测到可疑人员.</p>
                    <p>可能的描述: {single_caption}</p>
                    <p>现场图像：</p>
                    <p><img src="cid:alert_image"></p>
                    <p><small>此邮件由自动监控系统发送于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small></p>
                    """
                    send_frame_as_email(frame, msg)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 未检测到可疑人员。")
            

            
            self.last_process_time = time.time()
            self.process_count += 1
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 处理帧时出错: {e}")
        finally:
            self.process_running = False
            self.pending_process = False
    
    def emergency_process(self, frame, vqa_results):
        """处理紧急情况"""
        print(f"[{time.strftime('%H:%M:%S')}] 系统将在30秒内等待响应，否则将发送警报邮件。")
        print(f"[{time.strftime('%H:%M:%S')}] 请按任意键取消警报。")
        
        # 启动声音警报
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(CONFIG["emergency"]["alert_sound"])
            pygame.mixer.music.play(-1)
            print("声音警报已启动...")
        except pygame.error as e:
            print(f"[{time.strftime('%H:%M:%S')}] 无法播放声音警报: {e}")
        
        # 等待30秒或用户按键
        wait_time_seconds = 30
        start_time = time.time()
        key_pressed = False
        
        # 创建一个临时窗口用于按键检测
        cv2.imshow('Emergency Monitor', frame)
        cv2.resizeWindow("Emergency Monitor", 640, 480)
        
        while time.time() - start_time < wait_time_seconds:
            if cv2.waitKey(1) & 0xFF != 255:
                key_pressed = True
                break
        
        # 停止声音警报
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass

        cv2.destroyAllWindows()
        
        if key_pressed:
            print(f"[{time.strftime('%H:%M:%S')}] 用户已响应，警报已取消。")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 30秒内无响应，正在发送邮件警报...")
            mail_msg = f"""
            <p>监控系统检测到潜在的紧急情况，请立即查看！</p>
            <p>现场图像：</p>
            <p><img src="cid:alert_image"></p>
            <p><small>此邮件由自动监控系统发送于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small></p>
            """
            send_frame_as_email(frame, mail_msg)
            
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(CONFIG["emergency"]["succeed_sound"])
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except pygame.error as e:
                print(f"[{time.strftime('%H:%M:%S')}] 无法播放成功提示音: {e}")
    
    def process_frame(self, frame):
        """处理单帧"""
        current_time = time.time()
        
        # 休眠模式处理
        if self.is_sleeping:
            self.sleep_frame_counter += 1
            
            if self.sleep_frame_counter % self.sleep_frame_skip != 0:
                return {
                    'frame': frame,
                    'has_motion': False,
                    'contours': [],
                    'motion_area': 0,
                    'thresh': None,
                    'should_process': False,
                    'motion_boxes': [],
                    'is_sleeping': True,
                    'status': 'SLEEPING'
                }
            
            has_motion, contours, thresh, motion_area = self._detect_motion(frame, is_sleep_mode=True)
            self.sleep_frame_count += 1
            
            # 初始化稳定期检查
            if self.initialization_frames < self.initialization_threshold:
                self.initialization_frames += 1
                print(f"[{time.strftime('%H:%M:%S')}] 初始化中... {self.initialization_frames}/{self.initialization_threshold}")
                return {
                    'frame': frame,
                    'has_motion': False,
                    'contours': [],
                    'motion_area': 0,
                    'thresh': thresh,
                    'should_process': False,
                    'motion_boxes': [],
                    'is_sleeping': True,
                    'status': 'INITIALIZING'
                }
            
            # 检测到运动，唤醒系统
            if has_motion:
                self.is_sleeping = False
                self.wake_time = current_time
                self.wake_count += 1
                self.motion_start_time = current_time
                self.last_motion_time = current_time
                self.prev_frame = self.prev_sleep_frame
                print(f"[{time.strftime('%H:%M:%S')}] 检测到运动，系统唤醒！")
            
            return {
                'frame': frame,
                'has_motion': has_motion,
                'contours': contours,
                'motion_area': motion_area,
                'thresh': thresh,
                'should_process': False,
                'motion_boxes': [],
                'is_sleeping': True,
                'status': 'SLEEPING'
            }
        
        # 唤醒模式处理
        self.frame_count += 1
        
        has_motion, contours, thresh, motion_area = self._detect_motion(frame, is_sleep_mode=False)
        
        result = {
            'frame': frame,
            'has_motion': has_motion,
            'contours': contours,
            'motion_area': motion_area,
            'thresh': thresh,
            'should_process': False,
            'motion_boxes': [],
            'is_sleeping': False,
            'status': 'ACTIVE'
        }
        
        # 绘制运动框
        if has_motion:
            self.motion_frames += 1
            self.last_motion_time = current_time
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                result['motion_boxes'].append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if not self.is_motion_detected:
                self.is_motion_detected = True
                self.motion_start_time = current_time
            
            # 检查是否满足处理条件
            if self._should_process() and not self.process_running:
                motion_duration = current_time - self.motion_start_time if self.motion_start_time else 0
                
                if motion_duration >= self.motion_duration_threshold:
                    result['should_process'] = True
                    self.pending_process = True
                    print(f"[{time.strftime('%H:%M:%S')}] 检测到持续运动，准备处理帧")
        else:
            if self.is_motion_detected:
                self.is_motion_detected = False
                self.motion_start_time = None
            
            if self._should_sleep():
                self.is_sleeping = True
                self.prev_sleep_frame = self.prev_frame
                self.initialization_frames = 0
                print(f"[{time.strftime('%H:%M:%S')}] 长时间无运动，系统进入休眠模式")
                result['status'] = 'ENTERING_SLEEP'
        
        return result

class ModernSmartCalendar:
    def __init__(self, root):
        self.root = root
        self.root.title(CONFIG["window"]["title"])
        self.root.geometry(CONFIG["window"]["geometry"])
        self.root.configure(bg="#0f0f1e")
        
        # 设置窗口居中
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1600 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1600x900+{x}+{y}")
        
        # 初始化模型
        initialize_models()
        
        # 运动检测相关
        self.detector = MotionDetector(
            motion_threshold=CONFIG["motion_detector"]["motion_threshold"],
            min_contour_area=CONFIG["motion_detector"]["min_contour_area"],
            motion_duration_threshold=CONFIG["motion_detector"]["motion_duration_threshold"],
            sleep_timeout=CONFIG["motion_detector"]["sleep_timeout"],
            emergency_threshold=CONFIG["motion_detector"]["emergency_threshold"],
            emergency_cooldown=CONFIG["motion_detector"]["emergency_cooldown"]
        )
        
        # 摄像头相关
        self.cap = None
        self.camera_queue = queue.Queue()
        self.camera_thread = None
        self.camera_active = True
        self.camera_paused = False
        
        # 启动摄像头
        self.init_camera()
        
        # 自定义滚动条样式
        self.setup_scrollbar_style()
        
        # 创建渐变背景
        self.create_gradient_background()
        
        # 创建主容器
        self.main_container = tk.Frame(root, bg="#0f0f1e")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 顶部时间区域
        self.create_time_section()
        
        # 中间内容区域
        self.content_container = tk.Frame(self.main_container, bg="#0f0f1e")
        self.content_container.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # 左侧天气卡片
        self.create_weather_card()
        
        # 中间摄像头区域
        self.create_camera_section()
        
        # 右侧新闻图片区域
        self.create_news_section()
        
        # 底部状态栏
        self.create_status_bar()
        
        # 初始化数据
        self.update_time()
        self.update_weather()
        self.update_news()
        self.update_camera()
        
        # 定时更新
        self.root.after(CONFIG["update_intervals"]["time_ms"], self.update_time)
        self.root.after(CONFIG["update_intervals"]["weather_ms"], self.update_weather)
        self.root.after(CONFIG["update_intervals"]["news_ms"], self.update_news)
        
        # 绑定键盘事件
        self.root.bind('<KeyPress>', self.on_key_press)
        
        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print(f"[{time.strftime('%H:%M:%S')}] 错误：无法打开摄像头")
                self.camera_active = False
                return
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera"]["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera"]["height"])
            self.cap.set(cv2.CAP_PROP_FPS, CONFIG["camera"]["fps"])
            
            # 丢弃前几帧，让摄像头稳定
            print(f"[{time.strftime('%H:%M:%S')}] 摄像头预热中...")
            for _ in range(10):
                ret, _ = self.cap.read()
                if not ret:
                    time.sleep(0.1)
            
            # 启动摄像头线程
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            print(f"[{time.strftime('%H:%M:%S')}] 摄像头初始化成功")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 摄像头初始化失败: {e}")
            self.camera_active = False

    def setup_scrollbar_style(self):
        """设置自定义滚动条样式"""
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure(
            "Modern.Vertical.TScrollbar",
            troughcolor="#2a2a4e",
            background="#4a4a6e",
            bordercolor="#2a2a4e",
            lightcolor="#4a4a6e",
            darkcolor="#4a4a6e",
            arrowcolor="#2a2a4e",
            relief="flat",
            gripcount=0
        )
        
        style.map(
            "Modern.Vertical.TScrollbar",
            background=[('active', '#6a6a8e'), ('!active', '#4a4a6e')],
            troughcolor=[('active', '#2a2a4e'), ('!active', '#2a2a4e')]
        )

    def create_gradient_background(self):
        """创建渐变背景效果"""
        bg_frame = tk.Frame(self.root, bg="#0f0f1e")
        bg_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # 添加装饰性圆形
        for i in range(5):
            x = 0.1 + i * 0.2
            y = 0.8
            size = 200 - i * 20
            circle = tk.Frame(bg_frame, bg="#1a1a3e", width=size, height=size)
            circle.place(relx=x, rely=y, anchor="center")
            circle.tkraise()

    def create_time_section(self):
        """创建时间显示区域"""
        time_frame = tk.Frame(self.main_container, bg="#1a1a3e", relief=tk.FLAT, bd=0)
        time_frame.pack(fill=tk.X, pady=(0, 20))
        
        inner_frame = tk.Frame(time_frame, bg="#1a1a3e")
        inner_frame.pack(fill=tk.X, padx=30, pady=20)
        
        self.time_label = tk.Label(
            inner_frame,
            font=("Segoe UI", 56, "bold"),
            fg="#ffffff",
            bg="#1a1a3e"
        )
        self.time_label.pack()
        
        self.date_label = tk.Label(
            inner_frame,
            font=("Segoe UI", 18),
            fg="#a0a0c0",
            bg="#1a1a3e"
        )
        self.date_label.pack(pady=(5, 0))

    def create_weather_card(self):
        """创建天气卡片"""
        weather_container = tk.Frame(self.content_container, bg="#0f0f1e", width=350)
        weather_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        weather_container.pack_propagate(False)
        
        weather_card = tk.Frame(
            weather_container,
            bg="#2a2a4e",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        weather_card.pack(fill=tk.BOTH, expand=True)
        
        weather_inner = tk.Frame(weather_card, bg="#2a2a4e", padx=20, pady=20)
        weather_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        header_frame = tk.Frame(weather_inner, bg="#2a2a4e")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="🌤️ 今日天气",
            font=("Segoe UI", 18, "bold"),
            fg="#ffd700",
            bg="#2a2a4e"
        )
        title_label.pack()
        
        separator = tk.Frame(header_frame, bg="#4a4a6e", height=2)
        separator.pack(fill=tk.X, pady=(8, 0))
        
        info_frame = tk.Frame(weather_inner, bg="#2a2a4e")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.location_label = tk.Label(
            info_frame,
            font=("Segoe UI", 14),
            fg="#e0e0e0",
            bg="#2a2a4e"
        )
        self.location_label.pack(anchor="w", pady=(8, 4))
        
        self.temp_label = tk.Label(
            info_frame,
            font=("Segoe UI", 36, "bold"),
            fg="#ff6b6b",
            bg="#2a2a4e"
        )
        self.temp_label.pack(anchor="w", pady=(8, 4))
        
        self.condition_label = tk.Label(
            info_frame,
            font=("Segoe UI", 16),
            fg="#4ecdc4",
            bg="#2a2a4e"
        )
        self.condition_label.pack(anchor="w", pady=(8, 4))
        
        self.update_label = tk.Label(
            info_frame,
            font=("Segoe UI", 11),
            fg="#8080a0",
            bg="#2a2a4e"
        )
        self.update_label.pack(anchor="w", pady=(15, 0))

    def create_camera_section(self):
        """创建摄像头显示区域"""
        camera_container = tk.Frame(self.content_container, bg="#0f0f1e", width=450)
        camera_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=15)
        camera_container.pack_propagate(False)
        
        camera_card = tk.Frame(
            camera_container,
            bg="#2a2a4e",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        camera_card.pack(fill=tk.BOTH, expand=True)
        
        camera_inner = tk.Frame(camera_card, bg="#2a2a4e", padx=20, pady=20)
        camera_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        header_frame = tk.Frame(camera_inner, bg="#2a2a4e")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="📹 运动检测",
            font=("Segoe UI", 18, "bold"),
            fg="#ffd700",
            bg="#2a2a4e"
        )
        title_label.pack()
        
        separator = tk.Frame(header_frame, bg="#4a4a6e", height=2)
        separator.pack(fill=tk.X, pady=(8, 0))
        
        self.camera_label = tk.Label(
            camera_inner,
            bg="#000000",
            text="摄像头启动中...",
            font=("Segoe UI", 14),
            fg="#ffffff"
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        self.camera_status_label = tk.Label(
            camera_inner,
            text="状态: 休眠中",
            font=("Segoe UI", 12),
            fg="#a0a0c0",
            bg="#2a2a4e"
        )
        self.camera_status_label.pack(pady=(10, 0))

    def create_news_section(self):
        """创建新闻图片区域"""
        news_container = tk.Frame(self.content_container, bg="#0f0f1e")
        news_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))
        
        news_card = tk.Frame(
            news_container,
            bg="#2a2a4e",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        news_card.pack(fill=tk.BOTH, expand=True)
        
        news_inner = tk.Frame(news_card, bg="#2a2a4e", padx=20, pady=20)
        news_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        header_frame = tk.Frame(news_inner, bg="#2a2a4e")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="📰 今日要闻",
            font=("Segoe UI", 18, "bold"),
            fg="#ffd700",
            bg="#2a2a4e"
        )
        title_label.pack()
        
        separator = tk.Frame(header_frame, bg="#4a4a6e", height=2)
        separator.pack(fill=tk.X, pady=(8, 0))
        
        canvas_frame = tk.Frame(news_inner, bg="#2a2a4e")
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.news_canvas = tk.Canvas(
            canvas_frame,
            bg="#2a2a4e",
            highlightthickness=0
        )
        self.news_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient="vertical",
            command=self.news_canvas.yview,
            style="Modern.Vertical.TScrollbar"
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.news_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.news_inner_frame = tk.Frame(self.news_canvas, bg="#2a2a4e")
        self.news_canvas_window = self.news_canvas.create_window(
            (0, 0),
            window=self.news_inner_frame,
            anchor="nw"
        )
        
        self.news_label = tk.Label(
            self.news_inner_frame,
            bg="#2a2a4e",
            relief=tk.FLAT,
            bd=0
        )
        self.news_label.pack(pady=10)
        
        self.news_inner_frame.bind("<Configure>", self.on_news_frame_configure)
        self.news_canvas.bind("<Configure>", self.on_canvas_configure)

    def on_news_frame_configure(self, event):
        """更新滚动区域"""
        self.news_canvas.configure(scrollregion=self.news_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """调整内部框架宽度"""
        canvas_width = event.width
        self.news_canvas.itemconfig(self.news_canvas_window, width=canvas_width)

    def create_status_bar(self):
        """创建状态栏"""
        status_frame = tk.Frame(
            self.root,
            bg="#1a1a3e",
            height=40,
            relief=tk.FLAT,
            bd=0
        )
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            status_frame,
            text="🚀 智能台历已启动",
            font=("Segoe UI", 11),
            fg="#a0a0c0",
            bg="#1a1a3e"
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.timestamp_label = tk.Label(
            status_frame,
            text="",
            font=("Segoe UI", 10),
            fg="#707090",
            bg="#1a1a3e"
        )
        self.timestamp_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        control_label = tk.Label(
            status_frame,
            text="控制: [Q]退出 [W]唤醒 [S]休眠",
            font=("Segoe UI", 10),
            fg="#707090",
            bg="#1a1a3e"
        )
        control_label.pack(side=tk.RIGHT, padx=20, pady=10)

    def update_time(self):
        """更新时间显示"""
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y年%m月%d日 %A")
        
        self.time_label.config(text=time_str)
        self.date_label.config(text=date_str)
        self.timestamp_label.config(text=f"最后更新: {now.strftime('%H:%M')}")
        
        self.root.after(1000, self.update_time)

    def update_weather(self):
        """更新天气信息"""
        try:
            url = f"https://api.seniverse.com/v3/weather/now.json?key={CONFIG['api']['weather_api_key']}&location={CONFIG['api']['city_id']}&language=zh-Hans&unit=c"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if "results" in data:
                weather = data["results"][0]["now"]["text"]
                temperature = data["results"][0]["now"]["temperature"]
                location = data["results"][0]["location"]["name"]
                update_time = data["results"][0]["last_update"]
                
                self.location_label.config(text=f"📍 {location}")
                self.temp_label.config(text=f"{temperature}°C")
                self.condition_label.config(text=f"☁️ {weather}")
                self.update_label.config(text=f"🔄 更新时间: {update_time}")
                
                self.status_label.config(text="✅ 天气信息已更新")
            else:
                self.status_label.config(text="❌ 无法获取天气信息")
        except Exception as e:
            self.status_label.config(text=f"⚠️ 天气更新错误: {str(e)}")
        
        self.root.after(600000, self.update_weather)

    def update_news(self):
        """更新新闻图片"""
        try:
            response = requests.get("https://uapis.cn/api/v1/daily/news-image", timeout=10)
            if response.status_code == 200:
                image_data = io.BytesIO(response.content)
                image = Image.open(image_data)
                
                canvas_width = self.news_canvas.winfo_width()
                if canvas_width <= 1:
                    canvas_width = 700
                
                display_width = canvas_width - 40
                img_width, img_height = image.size
                
                ratio = display_width / img_width
                display_height = int(img_height * ratio)
                
                resized_image = image.resize(
                    (display_width, display_height),
                    Image.Resampling.LANCZOS
                )
                
                photo = ImageTk.PhotoImage(resized_image)
                self.news_label.config(image=photo)
                self.news_label.image = photo
                
                self.status_label.config(text="✅ 新闻图片已更新")
            else:
                self.status_label.config(text="❌ 无法获取新闻图片")
        except Exception as e:
            self.status_label.config(text=f"⚠️ 新闻更新错误: {str(e)}")
        
        self.root.after(3600000, self.update_news)
    
    def camera_loop(self):
        """摄像头线程循环"""
        while self.camera_active:
            if self.camera_paused:
                time.sleep(0.1)
                continue
                
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{time.strftime('%H:%M:%S')}] 错误：无法从摄像头读取帧。")
                time.sleep(0.1)
                continue
                
            # 处理帧
            result = self.detector.process_frame(frame)
            
            # 处理帧分析请求
            if result['should_process'] and not self.detector.process_running:
                threading.Thread(
                    target=self.detector.process_frame_with_models,
                    args=(frame,),
                    daemon=True
                ).start()
            
            # 将帧放入队列
            self.camera_queue.put(result)
    
    def update_camera(self):
        """更新摄像头显示"""
        try:
            while not self.camera_queue.empty():
                data = self.camera_queue.get_nowait()
                
                if isinstance(data, str):
                    if data == "PROCESSING":
                        self.camera_label.config(text="处理中...", image="")
                        self.camera_status_label.config(text="状态: 处理中")
                else:
                    frame = data['frame']
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image = image.resize((410, 308), Image.Resampling.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(image)
                    self.camera_label.config(image=photo, text="")
                    self.camera_label.image = photo
                    
                    status_text = "状态: "
                    if data['status'] == 'INITIALIZING':
                        status_text += "初始化中"
                    elif data['is_sleeping']:
                        status_text += "休眠中"
                    elif data['has_motion']:
                        status_text += f"检测到运动 (面积: {data['motion_area']:.0f})"
                    else:
                        status_text += "活跃"
                    
                    self.camera_status_label.config(text=status_text)
        except queue.Empty:
            pass
        
        self.root.after(30, self.update_camera)
    
    def on_key_press(self, event):
        """处理键盘按键事件"""
        key = event.keysym.lower()
        
        if key == 'q':
            self.on_closing()
        elif key == 'w':
            if self.detector.is_sleeping:
                self.detector.is_sleeping = False
                self.detector.wake_time = time.time()
                self.detector.last_motion_time = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] 手动唤醒系统")
        elif key == 's':
            if not self.detector.is_sleeping:
                self.detector.is_sleeping = True
                self.detector.prev_sleep_frame = self.detector.prev_frame
                self.detector.initialization_frames = 0
                print(f"[{time.strftime('%H:%M:%S')}] 手动进入休眠模式")
    
    def on_closing(self):
        """窗口关闭事件处理"""
        print(f"[{time.strftime('%H:%M:%S')}] 正在关闭程序...")
        self.camera_active = False
        self.camera_paused = True
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernSmartCalendar(root)
    root.mainloop()
