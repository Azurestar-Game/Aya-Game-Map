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
    # 告诉系统：我是高 DPI 应用程序，请给我返回真实的物理坐标
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

import win32gui  # 用于获取窗口句柄和位置
import keyboard  # 用于全局键盘监听

# --- 1. 配置区域 ---
pytesseract.pytesseract.tesseract_cmd = r'D:\Interface\Tesseract\tesseract.exe'

# 游戏窗口标题 (请修改为你游戏的实际标题，打开任务管理器可以看到)
GAME_WINDOW_TITLE = "这里填你的游戏窗口标题"

# OCR 区域 (这里需要填 *相对于游戏窗口左上角* 的偏移量)
# 之前测出的绝对坐标是 top:1220, left:793。
# 假设你当时游戏是全屏(1920x1080)或者位置在(0,0)，那这个就是相对坐标。
# 如果不确定，先保持这个，运行起来看看对不对。
OCR_OFFSET = {
    'top': 1057,
    'left': 20,
    'width': 180,
    'height': 17
}


# --- 2. 核心滤波类 (保持不变) ---
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
        self.rect = None

    def find_window(self):
        """查找包含指定标题的窗口"""

        def callback(hwnd, extra):
            title = win32gui.GetWindowText(hwnd)
            if self.title_pattern in title:
                self.hwnd = hwnd

        win32gui.EnumWindows(callback, None)
        return self.hwnd

    def get_rect(self):
        """获取窗口当前的屏幕绝对坐标 (left, top, width, height)"""
        if not self.hwnd:
            self.find_window()

        if self.hwnd:
            try:
                # rect 是 (left, top, right, bottom)
                rect = win32gui.GetWindowRect(self.hwnd)
                x, y = rect[0], rect[1]
                w = rect[2] - x
                h = rect[3] - y
                # 有些系统缩放可能导致这里不准，如果发现偏移，可能需要处理 DPI Awareness
                return {'left': x, 'top': y, 'width': w, 'height': h}
            except Exception as e:
                print(f"窗口句柄失效: {e}")
                self.hwnd = None
        return None

    def is_foreground(self):
        """判断该窗口是否处于当前激活状态（防止在聊天时误触）"""
        if not self.hwnd: return False
        return win32gui.GetForegroundWindow() == self.hwnd


# --- 4. 采集器主类 ---
class MapDataCollector:
    def __init__(self, base_output_dir=".", sampling_interval=0.5, game_title=""):
        self.sampling_interval = sampling_interval
        self.filter = TimeBasedFilter()

        # 初始化窗口管理器
        self.window_mgr = WindowManager(game_title)

        # 目录初始化
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = os.path.join(base_output_dir, f"map_data_{timestamp_str}")
        self.snapshots_dir = os.path.join(self.root_dir, "map_snapshots")
        self.csv_path = os.path.join(self.root_dir, "map_matrix.csv")
        self._initialize_structure()

        print(f"✅ 系统就绪。等待识别窗口: [{game_title}]")

    def _initialize_structure(self):
        os.makedirs(self.snapshots_dir, exist_ok=True)
        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'X', 'Y', 'Z', 'Image_Filename'])

    def _save_thumbnail(self, sct_img, filename_base):
        img = np.array(sct_img)
        max_width = 200
        height, width = img.shape[:2]
        if width > 0:  # 防止空图报错
            scaling_factor = max_width / float(width)
            new_height = int(height * scaling_factor)
            resized_img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
            img_name = f"{filename_base}.jpg"
            cv2.imwrite(os.path.join(self.snapshots_dir, img_name), resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return img_name
        return ""

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

            coord_str = str(coords) if coords else "Invalid"
            tag = "[Manual J]" if img_filename else "[Auto]"
            print(f"{tag} {coord_str} | {img_filename}")

    def run(self):
        with mss.mss() as sct:
            last_sample_time = time.time()

            # 等待找到窗口
            while not self.window_mgr.find_window():
                print("⏳ 正在搜索游戏窗口...", end='\r')
                time.sleep(1)
            print(f"\n✅ 已锁定窗口句柄: {self.window_mgr.hwnd}")

            while True:
                # 1. 获取当前窗口的实时位置 (支持拖动窗口)
                game_rect = self.window_mgr.get_rect()
                if not game_rect:
                    print("⚠️ 窗口丢失，重新搜索...")
                    time.sleep(1)
                    continue

                # 2. 计算 OCR 区域的绝对坐标
                # 绝对坐标 = 窗口左上角 + 相对偏移
                current_ocr_bbox = {
                    'top': game_rect['top'] + OCR_OFFSET['top'],
                    'left': game_rect['left'] + OCR_OFFSET['left'],
                    'width': OCR_OFFSET['width'],
                    'height': OCR_OFFSET['height']
                }

                # 3. 运行 OCR
                current_time = time.time()
                try:
                    ocr_sct = sct.grab(current_ocr_bbox)
                    ocr_img = np.array(ocr_sct)

                    scale_percent = 300
                    w = int(ocr_img.shape[1] * scale_percent / 100)
                    h = int(ocr_img.shape[0] * scale_percent / 100)
                    img_upscaled = cv2.resize(ocr_img, (w, h), interpolation=cv2.INTER_CUBIC)
                    hsv = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))

                    text = pytesseract.image_to_string(mask,
                                                       config=r'--psm 7 -c tessedit_char_whitelist=XYZ:,-0123456789').strip()

                    _, current_coords = self.filter.process(text)
                except Exception as e:
                    print(f"Capture Error: {e}")
                    current_coords = None

                # 4. 监听键盘 J (全局监听 + 焦点判断)
                # 使用 is_pressed 检测按键
                if keyboard.is_pressed('j'):
                    # 只有当游戏窗口是当前激活窗口时才触发
                    if self.window_mgr.is_foreground():
                        filename_base = f"{current_time:.3f}_manual"
                        # 抓取整个游戏窗口
                        game_sct = sct.grab(game_rect)
                        saved_name = self._save_thumbnail(game_sct, filename_base)
                        self._log_data(current_time, current_coords, saved_name)

                        # 简单的防抖动 (防止按一下存了10张)
                        time.sleep(0.3)

                # 按 Esc 退出脚本 (全局)
                if keyboard.is_pressed('esc'):
                    print("Exiting...")
                    break

                # 5. 自动采集逻辑
                if current_time - last_sample_time >= self.sampling_interval:
                    self._log_data(current_time, current_coords, img_filename="")
                    last_sample_time = current_time

                # 为了不让CPU跑满，加极小的sleep
                time.sleep(0.01)


if __name__ == "__main__":
    # ⚠️ 必须把这个改成你真正的游戏窗口名字，哪怕只是部分名字
    # 例如： "Minecraft", "Genshin", "World of Warcraft"
    TARGET_TITLE = "G2_depart"

    collector = MapDataCollector(
        base_output_dir="./game_data_output",
        sampling_interval=0.5,
        game_title=TARGET_TITLE
    )
    collector.run()