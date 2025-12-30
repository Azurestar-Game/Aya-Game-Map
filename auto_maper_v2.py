import time
import cv2
import mss
import numpy as np
import pytesseract
import re
import math
import os
import csv
from datetime import datetime
import pandas as pd
import ctypes

try:
    # 告诉系统：我是高 DPI 应用程序
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

import win32gui
import keyboard

# --- 1. 配置区域 ---
pytesseract.pytesseract.tesseract_cmd = r'D:\Interface\Tesseract\tesseract.exe'

# 游戏窗口标题
GAME_WINDOW_TITLE = "G2_depart"

# OCR 区域 (相对于窗口左上角)
OCR_OFFSET = {
    'top': 1057,
    'left': 20,
    'width': 180,
    'height': 17
}


# --- 2. 核心滤波类 ---
class TimeBasedFilter:
    def __init__(self, max_speed=15000, base_tolerance=1000):
        self.last_valid_pos = None
        self.last_valid_time = None
        self.max_speed = max_speed
        self.base_tolerance = base_tolerance
        self.min_coord_val = 1000

    def clean_data(self, raw_text):
        numbers = re.findall(r'-?\d+', raw_text)
        try:
            nums = [int(n) for n in numbers]
        except ValueError:
            return None
        if len(nums) > 3:
            try:
                if 2 in nums:
                    idx_2 = nums.index(2)
                    if 0 < idx_2 < len(nums) - 1: nums.pop(idx_2)
            except:
                pass
            if len(nums) > 3 and abs(nums[0]) < 100: nums.pop(0)
        if len(nums) != 3: return None
        return nums

    def process(self, raw_text):
        current_time = time.time()
        current_pos = self.clean_data(raw_text)
        if current_pos is None: return False, None
        if self.last_valid_pos is None:
            if abs(current_pos[0]) < self.min_coord_val: return False, None
            self.last_valid_pos = current_pos
            self.last_valid_time = current_time
            return True, current_pos
        dt = current_time - self.last_valid_time
        if dt <= 0: dt = 0.001
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.last_valid_pos, current_pos)))
        allowed_jump = (self.max_speed * dt) + self.base_tolerance
        if dist > allowed_jump:
            return False, None
        else:
            self.last_valid_pos = current_pos
            self.last_valid_time = current_time
            return True, current_pos


# --- 3. 窗口句柄工具 ---
class WindowManager:
    def __init__(self, title_pattern):
        self.title_pattern = title_pattern
        self.hwnd = None

    def find_window(self):
        def callback(hwnd, extra):
            title = win32gui.GetWindowText(hwnd)
            if self.title_pattern in title:
                self.hwnd = hwnd

        win32gui.EnumWindows(callback, None)
        return self.hwnd

    def get_rect(self):
        if not self.hwnd: self.find_window()
        if self.hwnd:
            try:
                rect = win32gui.GetWindowRect(self.hwnd)
                x, y = rect[0], rect[1]
                w = rect[2] - x
                h = rect[3] - y
                return {'left': x, 'top': y, 'width': w, 'height': h}
            except:
                self.hwnd = None
        return None

    def is_foreground(self):
        if not self.hwnd: return False
        return win32gui.GetForegroundWindow() == self.hwnd


# --- 4. 采集器主类 (修复版) ---
class MapDataCollector:
    def __init__(self, base_output_dir=".", sampling_interval=0.5, game_title=""):
        self.sampling_interval = sampling_interval
        self.filter = TimeBasedFilter()
        self.window_mgr = WindowManager(game_title)

        # 状态标志
        self.manual_snapshot_pending = False
        self.last_j_press_time = 0

        # 初始化目录
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = os.path.join(base_output_dir, f"map_data_{timestamp_str}")
        self.snapshots_dir = os.path.join(self.root_dir, "map_snapshots")
        self.csv_path = os.path.join(self.root_dir, "map_matrix.csv")
        self._initialize_structure()

        print(f"✅ 系统就绪。按 'J' 截图，按 '+' 退出。")

    def _initialize_structure(self):
        os.makedirs(self.snapshots_dir, exist_ok=True)
        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'X', 'Y', 'Z', 'Image_Filename'])

    def _save_thumbnail(self, sct_img, filename_base):
        # 保存原始分辨率截图，使用PNG无损格式
        img = np.array(sct_img)

        # 移除 Alpha 通道 (mss 截图默认是 BGRA，OpenCV 保存图片通常不需要 A 通道，除非你需要透明背景)
        # 如果保存出来的图颜色不对，或者你需要透明度，可以删掉下面这行
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # --- 核心修改点 ---
        # 1. 删除了 resize (缩放) 代码
        # 2. 将文件名后缀改为 .png (无损)
        # 3. 删除了 JPEG 压缩参数
        name = f"{filename_base}.png"
        save_path = os.path.join(self.snapshots_dir, name)

        # 保存图片 (PNG 默认就是无损的)
        cv2.imwrite(save_path, img)

        return name

    def _log_data(self, timestamp, coords, img_filename=""):
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [f"{timestamp:.3f}"]
            if coords:
                row.extend(coords)
            else:
                row.extend(['', '', ''])
            row.append(img_filename)
            writer.writerow(row)

            tag = "[J-SHOT]" if img_filename else "[Auto]"
            coord_str = str(coords) if coords else "Invalid"
            print(f"{tag} {coord_str}")

    def run(self):
        with mss.mss() as sct:
            last_sample_time = time.time()

            # 等待窗口
            while not self.window_mgr.find_window():
                print("⏳ 寻找窗口...", end='\r')
                time.sleep(1)
            print(f"\n✅ 锁定窗口: {self.window_mgr.hwnd}")

            while True:
                loop_start = time.time()
                game_rect = self.window_mgr.get_rect()

                if not game_rect:
                    print("⚠️ 窗口丢失")
                    time.sleep(1)
                    continue

                # --- 1. 按键监听 (非阻塞，极速响应) ---
                # 使用 '+' 号退出 (注意：keyboard 库中 + 号通常是 '+')
                if keyboard.is_pressed('+'):
                    print("\n🛑 用户停止采集")
                    break

                # 监听 J 键 (设置Pending标志位)
                # 只有在游戏窗口激活时才响应
                if keyboard.is_pressed('j'):
                    if self.window_mgr.is_foreground():
                        # 防抖动：距离上次按下至少 0.3秒
                        if loop_start - self.last_j_press_time > 0.3:
                            self.manual_snapshot_pending = True
                            self.last_j_press_time = loop_start
                            print(">>> J 键按下，等待 OCR 同步...")

                # --- 2. 图像采集与处理 ---
                # 计算 OCR 区域
                ocr_bbox = {
                    'top': game_rect['top'] + OCR_OFFSET['top'],
                    'left': game_rect['left'] + OCR_OFFSET['left'],
                    'width': OCR_OFFSET['width'],
                    'height': OCR_OFFSET['height']
                }

                # 抓取 OCR 图像
                try:
                    ocr_img_raw = np.array(sct.grab(ocr_bbox))

                    # 图像增强
                    scale = 300
                    h, w = ocr_img_raw.shape[:2]
                    upscaled = cv2.resize(ocr_img_raw, (w * scale // 100, h * scale // 100),
                                          interpolation=cv2.INTER_CUBIC)
                    hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))

                    # 识别
                    text = pytesseract.image_to_string(mask,
                                                       config=r'--psm 7 -c tessedit_char_whitelist=XYZ:,-0123456789').strip()
                    _, coords = self.filter.process(text)

                except Exception as e:
                    print(f"OCR Error: {e}")
                    coords = None

                # --- 3. 处理 J 键快照 (如果 Pending 为 True) ---
                if self.manual_snapshot_pending:
                    # 此时已经有了最新的 coords (即使是 None 也没关系，我们优先保图)

                    # 抓取全屏
                    try:
                        full_shot = sct.grab(game_rect)
                        fname_base = f"{loop_start:.3f}_manual"
                        saved_name = self._save_thumbnail(full_shot, fname_base)

                        # 保存数据
                        self._log_data(loop_start, coords, saved_name)

                        # 重置标志
                        self.manual_snapshot_pending = False

                    except Exception as e:
                        print(f"Snapshot Error: {e}")
                        self.manual_snapshot_pending = False

                # --- 4. 自动定时保存 ---
                elif loop_start - last_sample_time >= self.sampling_interval:
                    self._log_data(loop_start, coords, "")
                    last_sample_time = loop_start

                # 极短休眠防止死循环占满单核
                time.sleep(0.005)


if __name__ == "__main__":
    collector = MapDataCollector(
        base_output_dir="./game_data_output",
        sampling_interval=0.5,
        game_title="G2_depart"
    )
    collector.run()