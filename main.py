import cv2
import numpy as np
import time
import subprocess

class MotionDetector:
    """运动检测类，集成休眠唤醒机制和脚本执行功能"""
    
    def __init__(self, motion_threshold=1500, min_contour_area=500, 
                 motion_duration_threshold=0.5, sleep_timeout=10.0, 
                 emergency_threshold=5000, emergency_cooldown=30.0):
        """
        初始化运动检测器
        
        Args:
            motion_threshold: 运动检测阈值
            min_contour_area: 最小轮廓面积
            motion_duration_threshold: 持续运动时间阈值（秒）
            sleep_timeout: 无运动进入休眠的时间（秒）
            emergency_threshold: 紧急事件运动面积阈值
            emergency_cooldown: 紧急事件冷却时间（秒）
        """
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.motion_duration_threshold = motion_duration_threshold
        self.sleep_timeout = sleep_timeout
        self.emergency_threshold = emergency_threshold
        self.emergency_cooldown = emergency_cooldown
        
        # 初始化前一帧
        self.prev_frame = None
        self.prev_sleep_frame = None  # 休眠模式下的前一帧
        
        # 运动状态跟踪
        self.motion_start_time = None
        self.is_motion_detected = False
        self.last_motion_time = None  # 最后一次检测到运动的时间
        self.last_script_time = 0  # 最后一次执行脚本的时间
        self.script_interval = 5.0  # 脚本执行间隔（秒）
        
        # 休眠/唤醒状态
        self.is_sleeping = True  # 初始状态为休眠
        self.wake_time = None  # 唤醒时间
        
        # 窗口状态跟踪
        self.threshold_window_open = False  # 跟踪阈值窗口是否打开
        
        # 统计信息
        self.frame_count = 0
        self.sleep_frame_count = 0
        self.motion_frames = 0
        self.script_count = 0  # 脚本执行次数
        self.wake_count = 0  # 唤醒次数
        self.emergency_count = 0  # 紧急事件次数
        
        # 休眠模式参数
        self.sleep_frame_skip = 5  # 休眠模式下每5帧处理一次
        self.sleep_frame_counter = 0
        self.sleep_motion_threshold = 800  # 休眠模式下的运动阈值（更敏感）
        
        # 脚本执行状态
        self.script_running = False  # 标记脚本是否正在运行
        self.pending_script = False  # 标记是否有待执行的脚本
        
        # 紧急事件状态
        self.emergency_running = False
        self.last_emergency_time = 0  # 最后一次紧急事件时间
    
    def _preprocess_frame(self, frame):
        """预处理帧"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray
    
    def _detect_motion(self, frame, is_sleep_mode=False):
        """检测运动"""
        gray_frame = self._preprocess_frame(frame)
        
        # 选择对应的前一帧
        prev_frame = self.prev_sleep_frame if is_sleep_mode else self.prev_frame
        
        if prev_frame is None:
            if is_sleep_mode:
                self.prev_sleep_frame = gray_frame
            else:
                self.prev_frame = gray_frame
            return False, [], None, 0
        
        # 计算帧差
        diff = cv2.absdiff(prev_frame, gray_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # 检测轮廓
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓
        significant_contours = []
        total_motion_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = self.min_contour_area // 2 if is_sleep_mode else self.min_contour_area
            if area > min_area:
                significant_contours.append(contour)
                total_motion_area += area
        
        # 更新前一帧
        if is_sleep_mode:
            self.prev_sleep_frame = gray_frame
        else:
            self.prev_frame = gray_frame
        
        # 判断是否有显著运动
        threshold = self.sleep_motion_threshold if is_sleep_mode else self.motion_threshold
        has_motion = total_motion_area > threshold
        
        return has_motion, significant_contours, thresh, total_motion_area
    
    def _should_execute_script(self):
        """判断是否应该执行脚本"""
        current_time = time.time()
        return (current_time - self.last_script_time) >= self.script_interval
    
    def _should_sleep(self):
        """判断是否应该进入休眠"""
        if self.is_sleeping:
            return False
        
        if self.last_motion_time is None:
            return False
        
        current_time = time.time()
        return (current_time - self.last_motion_time) >= self.sleep_timeout
    
    def _safe_destroy_window(self, window_name):
        """安全销毁窗口"""
        try:
            # 检查窗口是否存在
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(window_name)
                return True
        except:
            pass
        return False
    
    def run_external_script(self):
        """执行外部阻塞脚本"""
        print(f"[{time.strftime('%H:%M:%S')}] 开始执行外部脚本...")
        self.script_running = True
        try:
            # 这里替换为您的实际脚本路径和参数
            # 例如：subprocess.run(["python", "your_script.py"], check=True)
            # 或者：subsystem.run(["/path/to/your/script"], check=True)
            subprocess.run(["python", "starting_main.py"], check=True)
            print(f"[{time.strftime('%H:%M:%S')}] 脚本执行完成")
            self.last_script_time = time.time()
            self.script_count += 1
            return True
        except subprocess.CalledProcessError as e:
            print(f"[{time.strftime('%H:%M:%S')}] 脚本执行错误: {e}")
            return False
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 脚本执行异常: {e}")
            return False
        finally:
            self.script_running = False
            self.pending_script = False
    
    def run_emergency_script(self):
        """执行紧急事件脚本"""
        print(f"[{time.strftime('%H:%M:%S')}] 检测到紧急事件！执行emergency.py...")
        self.emergency_running = True
        try:
            subprocess.run(["python", "emergency.py"], check=True)
            print(f"[{time.strftime('%H:%M:%S')}] 紧急事件处理完成")
            self.last_emergency_time = time.time()
            self.emergency_count += 1
            return True
        except subprocess.CalledProcessError as e:
            print(f"[{time.strftime('%H:%M:%S')}] 紧急事件处理错误: {e}")
            return False
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 紧急事件处理异常: {e}")
            return False
        finally:
            self.emergency_running = False
            # 重置帧缓存防止误触发
            self.prev_sleep_frame = None
            self.sleep_frame_counter = 0
    
    def process_frame(self, frame):
        """
        处理单帧
        
        Args:
            frame: 输入帧
            
        Returns:
            dict: 处理结果
        """
        current_time = time.time()
        
        # 休眠模式处理
        if self.is_sleeping:
            self.sleep_frame_counter += 1
            
            # 休眠模式下降低处理频率
            if self.sleep_frame_counter % self.sleep_frame_skip != 0:
                return {
                    'frame': frame,
                    'has_motion': False,
                    'contours': [],
                    'motion_area': 0,
                    'thresh': None,
                    'should_run_script': False,
                    'should_run_emergency': False,
                    'motion_boxes': [],
                    'is_sleeping': True,
                    'status': 'SLEEPING'
                }
            
            # 休眠模式下的运动检测
            has_motion, contours, thresh, motion_area = self._detect_motion(frame, is_sleep_mode=True)
            self.sleep_frame_count += 1
            
            # 检查紧急事件
            if has_motion and motion_area >= self.emergency_threshold:
                # 检查冷却时间
                if (current_time - self.last_emergency_time) >= self.emergency_cooldown:
                    print(f"[{time.strftime('%H:%M:%S')}] 检测到紧急运动（面积: {motion_area:.0f}）")
                    return {
                        'frame': frame,
                        'has_motion': has_motion,
                        'contours': contours,
                        'motion_area': motion_area,
                        'thresh': thresh,
                        'should_run_script': False,
                        'should_run_emergency': True,
                        'motion_boxes': [],
                        'is_sleeping': True,
                        'status': 'EMERGENCY'
                    }
            
            # 检测到运动，唤醒系统
            if has_motion:
                self.is_sleeping = False
                self.wake_time = current_time
                self.wake_count += 1
                self.motion_start_time = current_time
                self.last_motion_time = current_time
                self.prev_frame = self.prev_sleep_frame  # 继承休眠时的前一帧
                print(f"[{time.strftime('%H:%M:%S')}] 检测到运动，系统唤醒！")
            
            return {
                'frame': frame,
                'has_motion': has_motion,
                'contours': contours,
                'motion_area': motion_area,
                'thresh': thresh,
                'should_run_script': False,
                'should_run_emergency': False,
                'motion_boxes': [],
                'is_sleeping': True,
                'status': 'SLEEPING'
            }
        
        # 唤醒模式处理
        self.frame_count += 1
        
        # 检测运动
        has_motion, contours, thresh, motion_area = self._detect_motion(frame, is_sleep_mode=False)
        
        result = {
            'frame': frame,
            'has_motion': has_motion,
            'contours': contours,
            'motion_area': motion_area,
            'thresh': thresh,
            'should_run_script': False,
            'should_run_emergency': False,
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
            
            # 更新运动状态
            if not self.is_motion_detected:
                self.is_motion_detected = True
                self.motion_start_time = current_time
            
            # 检查是否满足触发条件
            if self._should_execute_script():
                motion_duration = current_time - self.motion_start_time if self.motion_start_time else 0
                
                if motion_duration >= self.motion_duration_threshold and not self.script_running:
                    result['should_run_script'] = True
                    self.pending_script = True
                    print(f"[{time.strftime('%H:%M:%S')}] 检测到持续运动，准备执行脚本")
        else:
            # 重置运动状态
            if self.is_motion_detected:
                self.is_motion_detected = False
                self.motion_start_time = None
            
            # 检查是否应该进入休眠
            if self._should_sleep():
                self.is_sleeping = True
                self.prev_sleep_frame = self.prev_frame  # 保存当前帧作为休眠的起始帧
                print(f"[{time.strftime('%H:%M:%S')}] 长时间无运动，系统进入休眠模式")
                result['status'] = 'ENTERING_SLEEP'
        
        return result
    
    def get_statistics(self):
        """获取统计信息"""
        total_frames = self.frame_count + self.sleep_frame_count
        return {
            'total_frames': total_frames,
            'active_frames': self.frame_count,
            'sleep_frames': self.sleep_frame_count,
            'motion_frames': self.motion_frames,
            'scripts_executed': self.script_count,
            'emergency_triggered': self.emergency_count,
            'wake_count': self.wake_count,
            'motion_ratio': self.motion_frames / max(self.frame_count, 1) * 100 if self.frame_count > 0 else 0,
            'sleep_ratio': self.sleep_frame_count / max(total_frames, 1) * 100,
            'current_status': 'SLEEPING' if self.is_sleeping else 'ACTIVE'
        }


def main():
    """主函数 - 运动检测触发脚本执行，支持休眠唤醒"""
    print("=== 运动检测触发脚本执行系统（支持休眠唤醒）===")
    
    # 初始化运动检测器
    detector = MotionDetector(
        motion_threshold=400,              # 运动面积阈值
        min_contour_area=100,               # 最小轮廓面积
        motion_duration_threshold=0.5,      # 持续运动时间阈值（秒）
        sleep_timeout=120.0,                # 10秒无运动进入休眠
        emergency_threshold=5000,           # 紧急事件运动面积阈值
        emergency_cooldown=30.0             # 紧急事件冷却时间（秒）
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已启动，系统处于休眠模式，等待运动唤醒...")
    print("按 'q' 键退出")
    print("按 'w' 键手动唤醒")
    print("按 's' 键手动休眠")
    print("-" * 50)
    
    # 显示窗口
    cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion Detection", 640, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        result = detector.process_frame(frame)
        
        # 处理紧急事件
        if result['should_run_emergency'] and not detector.emergency_running:
            # 释放摄像头资源
            cap.release()
            cv2.destroyAllWindows()
            
            # 执行紧急脚本
            detector.run_emergency_script()
            
            # 重新初始化摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("错误：无法重新打开摄像头")
                break
                
            # 重新创建窗口
            cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
            print(f"[{time.strftime('%H:%M:%S')}] 系统已恢复运动检测（休眠模式）")
            continue  # 跳过当前帧处理
        
        # 根据状态显示不同的信息
        if result['is_sleeping']:
            # 休眠模式显示
            status_color = (0, 0, 255)  # 红色
            status_text = "SLEEPING - Waiting for motion..."
            info_text = f"Sleep frames: {detector.sleep_frame_count}"
            
            # 在休眠模式下添加半透明遮罩
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        else:
            # 唤醒模式显示
            status_color = (0, 255, 0)  # 绿色
            status_text = "ACTIVE"
            info_text = f"Moving: {'yes' if result['has_motion'] else 'no'} | Area: {result['motion_area']:.0f}"
            if detector.pending_script:
                info_text += " | Script pending..."
            if detector.script_running:
                info_text += " | Script running..."
        
        # 显示状态信息
        cv2.putText(result['frame'], status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(result['frame'], info_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示倒计时（如果在唤醒模式）
        if not result['is_sleeping'] and detector.last_motion_time:
            time_until_sleep = detector.sleep_timeout - (time.time() - detector.last_motion_time)
            if time_until_sleep > 0:
                cv2.putText(result['frame'], f"Sleep in: {time_until_sleep:.1f}s", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 显示紧急事件冷却倒计时
        if detector.last_emergency_time > 0:
            cooldown_remaining = detector.emergency_cooldown - (time.time() - detector.last_emergency_time)
            if cooldown_remaining > 0:
                cv2.putText(result['frame'], f"Emergency cooldown: {cooldown_remaining:.1f}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 显示结果
        cv2.imshow("Motion Detection", result['frame'])
        
        # 处理阈值窗口
        if not result['is_sleeping'] and result['thresh'] is not None:
            if not detector.threshold_window_open:
                cv2.namedWindow("Motion Threshold", cv2.WINDOW_NORMAL)
                detector.threshold_window_open = True
            cv2.imshow("Motion Threshold", result['thresh'])
        elif result['is_sleeping']:
            # 安全销毁阈值窗口
            if detector.threshold_window_open:
                detector._safe_destroy_window("Motion Threshold")
                detector.threshold_window_open = False
        
        # 当需要执行脚本时
        if result['should_run_script'] and not detector.script_running:
            # 释放摄像头资源
            cap.release()
            cv2.destroyAllWindows()
            
            # 执行阻塞脚本
            script_success = detector.run_external_script()
            
            # 重新初始化摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("错误：无法重新打开摄像头")
                break
                
            # 重置检测器状态
            detector.prev_frame = None
            detector.prev_sleep_frame = None
            detector.last_motion_time = time.time()
            detector.motion_start_time = None
            detector.is_motion_detected = False
            
            # 重新创建窗口
            cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
            print(f"[{time.strftime('%H:%M:%S')}] 系统已恢复运动检测")
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):  # 手动唤醒
            if detector.is_sleeping:
                detector.is_sleeping = False
                detector.wake_time = time.time()
                detector.last_motion_time = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] 手动唤醒系统")
        elif key == ord('s'):  # 手动休眠
            if not detector.is_sleeping:
                detector.is_sleeping = True
                detector.prev_sleep_frame = detector.prev_frame
                print(f"[{time.strftime('%H:%M:%S')}] 手动进入休眠模式")
    
    # 清理资源
    cap.release()
    
    # 安全销毁所有窗口
    detector._safe_destroy_window("Motion Detection")
    detector._safe_destroy_window("Motion Threshold")
    cv2.destroyAllWindows()
    
    # 显示最终统计
    final_stats = detector.get_statistics()
    print("\n=== 最终统计 ===")
    print(f"总帧数: {final_stats['total_frames']}")
    print(f"  - 活跃帧数: {final_stats['active_frames']}")
    print(f"  - 休眠帧数: {final_stats['sleep_frames']}")
    print(f"运动帧数: {final_stats['motion_frames']}")
    print(f"执行脚本数: {final_stats['scripts_executed']}")
    print(f"紧急事件数: {final_stats['emergency_triggered']}")
    print(f"唤醒次数: {final_stats['wake_count']}")
    print(f"运动比例: {final_stats['motion_ratio']:.1f}%")
    print(f"休眠比例: {final_stats['sleep_ratio']:.1f}%")
    print(f"最终状态: {final_stats['current_status']}")
    print("程序已结束")

if __name__ == "__main__":
    main()
