import requests
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
from datetime import datetime
import time

# API配置
WEATHER_API_KEY = "SWOiSWkZWBrzXEjHx"  # 替换为你的天气API密钥
CITY_ID = "东莞"
NEWS_API_URL = "https://uapis.cn/api/v1/daily/news-image"

class ModernSmartCalendar:
    def __init__(self, root):
        self.root = root
        self.root.title("智能台历 - 每日天气与新闻")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0f0f1e")
        
        # 设置窗口居中
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")
        
        # 自定义滚动条样式
        self.setup_scrollbar_style()
        
        # 创建渐变背景
        self.create_gradient_background()
        
        # 创建主容器
        self.main_container = tk.Frame(root, bg="#0f0f1e")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # 顶部时间区域 - 玻璃态效果
        self.create_time_section()
        
        # 中间内容区域
        self.content_container = tk.Frame(self.main_container, bg="#0f0f1e")
        self.content_container.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # 左侧天气卡片
        self.create_weather_card()
        
        # 右侧新闻图片区域 - 优化为竖向布局
        self.create_news_section()
        
        # 底部状态栏
        self.create_status_bar()
        
        # 初始化数据
        self.update_time()
        self.update_weather()
        self.update_news()
        
        # 定时更新
        self.root.after(1000, self.update_time)
        self.root.after(600000, self.update_weather)
        self.root.after(3600000, self.update_news)

    def setup_scrollbar_style(self):
        """设置自定义滚动条样式"""
        style = ttk.Style()
        style.theme_use('default')
        
        # 配置滚动条样式
        style.configure(
            "Modern.Vertical.TScrollbar",
            troughcolor="#2a2a4e",  # 滑轨颜色与背景融合
            background="#4a4a6e",   # 滑动块颜色
            bordercolor="#2a2a4e",  # 边框颜色与背景融合
            lightcolor="#4a4a6e",
            darkcolor="#4a4a6e",
            arrowcolor="#2a2a4e",   # 箭头颜色透明
            relief="flat",           # 无边框
            gripcount=0              # 无抓取点
        )
        
        # 配置滑动块样式
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
        
        # 添加内边距
        inner_frame = tk.Frame(time_frame, bg="#1a1a3e")
        inner_frame.pack(fill=tk.X, padx=30, pady=20)
        
        # 时间标签
        self.time_label = tk.Label(
            inner_frame,
            font=("Segoe UI", 56, "bold"),
            fg="#ffffff",
            bg="#1a1a3e"
        )
        self.time_label.pack()
        
        # 日期标签
        self.date_label = tk.Label(
            inner_frame,
            font=("Segoe UI", 18),
            fg="#a0a0c0",
            bg="#1a1a3e"
        )
        self.date_label.pack(pady=(5, 0))

    def create_weather_card(self):
        """创建天气卡片"""
        weather_container = tk.Frame(self.content_container, bg="#0f0f1e")
        weather_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        
        # 天气卡片主体
        weather_card = tk.Frame(
            weather_container,
            bg="#2a2a4e",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        weather_card.pack(fill=tk.BOTH, expand=True)
        
        # 添加圆角效果（通过内边距模拟）
        weather_inner = tk.Frame(weather_card, bg="#2a2a4e", padx=25, pady=25)
        weather_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 天气图标和标题
        header_frame = tk.Frame(weather_inner, bg="#2a2a4e")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="🌤️ 今日天气",
            font=("Segoe UI", 20, "bold"),
            fg="#ffd700",
            bg="#2a2a4e"
        )
        title_label.pack()
        
        # 分隔线
        separator = tk.Frame(header_frame, bg="#4a4a6e", height=2)
        separator.pack(fill=tk.X, pady=(10, 0))
        
        # 天气信息区域
        info_frame = tk.Frame(weather_inner, bg="#2a2a4e")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        # 位置
        self.location_label = tk.Label(
            info_frame,
            font=("Segoe UI", 16),
            fg="#e0e0e0",
            bg="#2a2a4e"
        )
        self.location_label.pack(anchor="w", pady=(10, 5))
        
        # 温度 - 大字体显示
        self.temp_label = tk.Label(
            info_frame,
            font=("Segoe UI", 42, "bold"),
            fg="#ff6b6b",
            bg="#2a2a4e"
        )
        self.temp_label.pack(anchor="w", pady=(10, 5))
        
        # 天气状况
        self.condition_label = tk.Label(
            info_frame,
            font=("Segoe UI", 18),
            fg="#4ecdc4",
            bg="#2a2a4e"
        )
        self.condition_label.pack(anchor="w", pady=(10, 5))
        
        # 更新时间
        self.update_label = tk.Label(
            info_frame,
            font=("Segoe UI", 12),
            fg="#8080a0",
            bg="#2a2a4e"
        )
        self.update_label.pack(anchor="w", pady=(20, 0))

    def create_news_section(self):
        """创建新闻图片区域 - 优化竖向图片显示"""
        news_container = tk.Frame(self.content_container, bg="#0f0f1e")
        news_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))
        
        # 新闻卡片
        news_card = tk.Frame(
            news_container,
            bg="#2a2a4e",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        news_card.pack(fill=tk.BOTH, expand=True)
        
        # 内边距
        news_inner = tk.Frame(news_card, bg="#2a2a4e", padx=25, pady=25)
        news_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 标题区域
        header_frame = tk.Frame(news_inner, bg="#2a2a4e")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="📰 今日要闻",
            font=("Segoe UI", 20, "bold"),
            fg="#ffd700",
            bg="#2a2a4e"
        )
        title_label.pack()
        
        # 分隔线
        separator = tk.Frame(header_frame, bg="#4a4a6e", height=2)
        separator.pack(fill=tk.X, pady=(10, 0))
        
        # 图片显示区域 - 使用Canvas实现滚动
        canvas_frame = tk.Frame(news_inner, bg="#2a2a4e")
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.news_canvas = tk.Canvas(
            canvas_frame,
            bg="#2a2a4e",
            highlightthickness=0,
            width=400,
            height=500
        )
        self.news_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 自定义滚动条
        scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient="vertical",
            command=self.news_canvas.yview,
            style="Modern.Vertical.TScrollbar"
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.news_canvas.configure(yscrollcommand=scrollbar.set)
        
        # 创建可滚动的内部框架
        self.news_inner_frame = tk.Frame(self.news_canvas, bg="#2a2a4e")
        self.news_canvas_window = self.news_canvas.create_window(
            (0, 0),
            window=self.news_inner_frame,
            anchor="nw"
        )
        
        # 新闻图片标签
        self.news_label = tk.Label(
            self.news_inner_frame,
            bg="#2a2a4e",
            relief=tk.FLAT,
            bd=0
        )
        self.news_label.pack(pady=10)
        
        # 绑定滚动事件
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
        
        # 状态标签
        self.status_label = tk.Label(
            status_frame,
            text="🚀 智能台历已启动",
            font=("Segoe UI", 11),
            fg="#a0a0c0",
            bg="#1a1a3e"
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # 时间戳
        self.timestamp_label = tk.Label(
            status_frame,
            text="",
            font=("Segoe UI", 10),
            fg="#707090",
            bg="#1a1a3e"
        )
        self.timestamp_label.pack(side=tk.RIGHT, padx=20, pady=10)

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
            url = f"https://api.seniverse.com/v3/weather/now.json?key={WEATHER_API_KEY}&location={CITY_ID}&language=zh-Hans&unit=c"
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
            response = requests.get(NEWS_API_URL, timeout=10)
            if response.status_code == 200:
                image_data = io.BytesIO(response.content)
                image = Image.open(image_data)
                
                # 计算合适的显示宽度（考虑canvas宽度）
                display_width = 680  # 留一些边距
                img_width, img_height = image.size
                
                # 保持宽高比计算高度
                ratio = display_width / img_width
                display_height = int(img_height * ratio)
                
                # 调整图片大小
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernSmartCalendar(root)
    root.mainloop()
