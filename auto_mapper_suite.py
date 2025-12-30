import matplotlib

# 强制使用 TkAgg 后端，避免 Qt 冲突
try:
    matplotlib.use('TkAgg')
except:
    pass

import time
import cv2
import mss
import numpy as np
import pytesseract
import re
import math
import os
import csv
import threading
import pandas as pd
import ctypes
import win32gui
import keyboard
from datetime import datetime
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

# 【修复 2】: 设置中文字体
# Windows 系统通常使用 'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider

# ==========================================
# 1. 全局配置区域
# ==========================================
# ⚠️ 请修改为你的 Tesseract 路径
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

# 运行参数
SAMPLING_INTERVAL = 0.1  # OCR 采集间隔 (秒)
AUTO_FIX_INTERVAL = 10000  # 自动刷新间隔 (毫秒)

# 高 DPI 适配
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

# CSV 读写锁
csv_lock = threading.Lock()


# ==========================================
# 模块一：核心算法 (Filter & Fixer)
# ==========================================

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


class TrajectoryRepairerV4:
    def __init__(self, df):
        self.df = df.sort_values('Timestamp').reset_index(drop=True)
        self.outlier_threshold = 3000
        self.window_size = 15

    def generate_candidates(self, val):
        if pd.isna(val): return []
        val_int = int(val)
        val_str = str(abs(val_int))
        candidates = {val_int, -val_int, int(val_int / 10), int(val_int * 10)}
        if len(val_str) > 2:
            try:
                sign = 1 if val_int >= 0 else -1
                candidates.add(sign * int(val_str[1:]))
                candidates.add(int(val_int // 10))
            except:
                pass
        return list(candidates)

    def fix_axis(self, series_vals):
        vals = series_vals.values
        n = len(vals)
        fixed_vals = np.copy(vals)
        series_pd = pd.Series(vals)
        median_guide = series_pd.rolling(window=self.window_size, center=True, min_periods=1).median().values
        last_valid_val = median_guide[0] if len(median_guide) > 0 else 0

        for i in range(n):
            raw = vals[i]
            guide = median_guide[i]
            if pd.isna(raw):
                fixed_vals[i] = last_valid_val
                continue

            diff_guide = abs(raw - guide)
            candidates = self.generate_candidates(raw)
            best_cand = raw

            if diff_guide > self.outlier_threshold:
                min_dist = float('inf')
                for cand in candidates:
                    d = abs(cand - guide)
                    if d < min_dist:
                        min_dist = d
                        best_cand = cand
                if min_dist > self.outlier_threshold:
                    best_cand = guide

            dist_to_last = abs(best_cand - last_valid_val)
            dist_guide_to_last = abs(guide - last_valid_val)

            if dist_guide_to_last < (self.outlier_threshold / 2) and dist_to_last > self.outlier_threshold:
                final_val = last_valid_val
            else:
                final_val = best_cand

            fixed_vals[i] = final_val
            last_valid_val = final_val
        return fixed_vals

    def run(self):
        fixed_data = {}
        for axis in ['X', 'Y', 'Z']:
            if axis not in self.df.columns: continue
            raw_series = self.df[axis].interpolate(method='linear', limit_direction='both')
            fixed_data[axis] = self.fix_axis(raw_series)

        df_fixed = self.df.copy()
        for axis in ['X', 'Y', 'Z']:
            if axis in fixed_data:
                df_fixed[axis] = fixed_data[axis]
        return df_fixed


# ==========================================
# 模块二：数据采集器 (后台线程)
# ==========================================

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


class MapDataCollector(threading.Thread):
    def __init__(self, base_output_dir):
        super().__init__()
        self.daemon = True
        self.running = True

        self.filter = TimeBasedFilter()
        self.window_mgr = WindowManager(GAME_WINDOW_TITLE)

        self.manual_snapshot_pending = False
        self.last_j_press_time = 0

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = os.path.join(base_output_dir, f"map_data_{timestamp_str}")
        self.snapshots_dir = os.path.join(self.root_dir, "map_snapshots")
        self.csv_path = os.path.join(self.root_dir, "map_matrix.csv")

        os.makedirs(self.snapshots_dir, exist_ok=True)
        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'X', 'Y', 'Z', 'Image_Filename'])

        print(f"✅ [Collector] Saving to: {self.root_dir}")

    def save_snapshot_raw(self, sct_img, filename_base):
        img = np.array(sct_img)
        name = f"{filename_base}.jpg"
        path = os.path.join(self.snapshots_dir, name)
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return name

    def log_data(self, timestamp, coords, img_filename=""):
        with csv_lock:
            with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [f"{timestamp:.3f}"]
                if coords:
                    row.extend(coords)
                else:
                    row.extend(['', '', ''])
                row.append(img_filename)
                writer.writerow(row)

        if img_filename:
            print(f"[📸 J-SHOT] {coords}")

    def run(self):
        print("✅ [Collector] Thread Started...")
        with mss.mss() as sct:
            last_sample_time = time.time()

            while not self.window_mgr.find_window() and self.running:
                time.sleep(1)

            if self.running:
                print(f"✅ [Collector] Locked Window: {self.window_mgr.hwnd}")

            while self.running:
                loop_start = time.time()
                game_rect = self.window_mgr.get_rect()

                if not game_rect:
                    time.sleep(1)
                    continue

                if keyboard.is_pressed('+'):
                    print("\n🛑 [Command] '+' pressed. Exiting...")
                    self.running = False
                    break

                if keyboard.is_pressed('j'):
                    if self.window_mgr.is_foreground():
                        if loop_start - self.last_j_press_time > 0.3:
                            self.manual_snapshot_pending = True
                            self.last_j_press_time = loop_start
                            print(">>> J Pressed. Snapshot pending...")

                # OCR
                ocr_bbox = {
                    'top': game_rect['top'] + OCR_OFFSET['top'],
                    'left': game_rect['left'] + OCR_OFFSET['left'],
                    'width': OCR_OFFSET['width'],
                    'height': OCR_OFFSET['height']
                }

                coords = None
                try:
                    ocr_img = np.array(sct.grab(ocr_bbox))
                    h, w = ocr_img.shape[:2]
                    scale = 300
                    upscaled = cv2.resize(ocr_img, (w * scale // 100, h * scale // 100), interpolation=cv2.INTER_CUBIC)
                    hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
                    text = pytesseract.image_to_string(mask,
                                                       config=r'--psm 7 -c tessedit_char_whitelist=XYZ:,-0123456789').strip()
                    _, coords = self.filter.process(text)
                except:
                    pass

                # Snapshot
                if self.manual_snapshot_pending:
                    try:
                        full_shot = sct.grab(game_rect)
                        fname = f"{loop_start:.3f}_manual"
                        saved_name = self.save_snapshot_raw(full_shot, fname)
                        self.log_data(loop_start, coords, saved_name)
                    except Exception as e:
                        print(f"Snapshot Failed: {e}")
                    finally:
                        self.manual_snapshot_pending = False

                # Auto Log
                elif loop_start - last_sample_time >= SAMPLING_INTERVAL:
                    self.log_data(loop_start, coords, "")
                    last_sample_time = loop_start

                time.sleep(0.005)


# ==========================================
# 模块三：可视化 (主线程 GUI)
# ==========================================

class EngineeringVisualizer(ABC):
    def __init__(self, collector_instance):
        self.collector = collector_instance
        self.folder_path = self.collector.root_dir
        self.snapshots_dir = self.collector.snapshots_dir

        self.raw_csv = self.collector.csv_path
        self.fixed_csv = os.path.join(self.folder_path, "map_matrix_fixed.csv")

        self.df = None
        self.fig = None

        self.setup_ui()

    def run_fixer(self):
        if not os.path.exists(self.raw_csv): return False

        try:
            with csv_lock:
                if os.path.getsize(self.raw_csv) < 50: return False
                df_raw = pd.read_csv(self.raw_csv)

            if df_raw.empty or len(df_raw) < 5: return False

            repairer = TrajectoryRepairerV4(df_raw)
            df_fixed = repairer.run()
            df_fixed.to_csv(self.fixed_csv, index=False)
            return True
        except Exception as e:
            return False

    def load_data_for_viz(self):
        target_csv = self.fixed_csv if os.path.exists(self.fixed_csv) else self.raw_csv
        if not os.path.exists(target_csv): return False

        try:
            raw_df = pd.read_csv(target_csv)
            if raw_df.empty: return False

            # 1. 基础过滤：保证坐标有效
            # 兼容列名大小写或空格问题（可选，但推荐检查）
            # 这里假设列名就是严格的 'X', 'Y', 'Z'
            is_valid = raw_df['X'].notna() & raw_df['Y'].notna() & raw_df['Z'].notna()

            # 2. 计算拓扑分段
            raw_df['segment_id'] = (~is_valid).astype(int).cumsum()

            # 3. 提取有效数据
            self.df = raw_df[is_valid].copy()

            # 强制转换坐标为数值类型 (防止里面混入字符串导致后续运算崩溃)
            for col in ['X', 'Y', 'Z']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')  # coerce会将无法转数字的变成NaN

            # ================= 核心修复部分 =================
            # 修复逻辑：
            # 1. 检查列是否存在：防止CSV里根本没这个字段导致 KeyError
            # 2. fillna(""): 处理空值
            # 3. astype(str): 【关键】强制转字符串，防止数字类型导致 .str 报错
            if 'Image_Filename' in self.df.columns:
                self.df['HasImage'] = self.df['Image_Filename'].fillna("").astype(str).str.strip() != ""
            else:
                # 如果CSV里没有这一列，默认全为False
                self.df['HasImage'] = False
            # ==============================================

            # 4. 转 NumPy (加速)
            self.np_x = self.df['X'].values
            self.np_y = self.df['Y'].values
            self.np_z = self.df['Z'].values
            self.np_has_img = self.df['HasImage'].values.astype(bool)  # 确保是布尔型
            self.np_seg_id = self.df['segment_id'].values

            # 5. 计算连通性掩码
            if len(self.np_seg_id) > 1:
                self.connect_mask = (self.np_seg_id[:-1] == self.np_seg_id[1:])
            else:
                self.connect_mask = np.array([], dtype=bool)

            # 6. 全局范围计算
            if len(self.np_z) > 0:
                self.z_min_global = self.np_z.min()
                self.z_max_global = self.np_z.max()
            else:
                self.z_min_global = 0.0
                self.z_max_global = 100.0

            return True

        except Exception as e:
            # 强烈建议把 e 打印出来，否则出错不知道错在哪
            print(f"❌ 数据解析异常 ({target_csv}): {e}")
            return False

    def setup_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Real-time Map Visualizer")
        self.fig.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.3)

        # 布局
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Top
        self.ax2 = self.fig.add_subplot(2, 2, 3)  # Side (Profile)
        self.ax3 = self.fig.add_subplot(1, 2, 2, projection='3d')  # 3D

        COLOR_PATH = '#555555'
        COLOR_STAR = '#FF0000'
        CMAP = 'viridis'

        # --- 1. Top View (XY) ---
        self.lc1 = LineCollection([], colors=COLOR_PATH, lw=0.8, alpha=0.4)
        self.ax1.add_collection(self.lc1)
        self.sc1 = self.ax1.scatter([], [], c=[], cmap=CMAP, s=15, alpha=0.7)
        self.st1 = self.ax1.scatter([], [], c=COLOR_STAR, marker='*', s=150, zorder=100)
        self.ax1.set_title("1. Top View (XY - North Up)")
        self.ax1.grid(True, alpha=0.3)
        # 【关键修改】强制等比例，防止被拉伸
        self.ax1.set_aspect('equal', adjustable='datalim')

        # --- 2. Profile (Side View) ---
        self.lc2 = LineCollection([], colors=COLOR_PATH, lw=0.8, alpha=0.4)
        self.ax2.add_collection(self.lc2)
        self.sc2 = self.ax2.scatter([], [], c=[], cmap=CMAP, s=15, alpha=0.7)
        self.ax2.set_title("2. Profile Projection")
        self.ax2.grid(True, linestyle='--')
        # 【关键修改】侧视图也建议等比例，否则坡度看起来会失真
        self.ax2.set_aspect('equal', adjustable='datalim')

        # 增加一个文本对象用于显示具体的方位角 (如 "北偏东 27.3°")
        self.txt_angle_label = self.fig.text(0.15, 0.13, "Direction: N", fontsize=10, color='blue', fontweight='bold')

        # --- 3. 3D View ---
        dummy_seg = np.array([[[0, 0, 0], [0, 0, 0]]])
        self.lc3 = Line3DCollection(dummy_seg, colors=COLOR_PATH, lw=0.5, alpha=0.3)
        self.ax3.add_collection3d(self.lc3)
        self.sc3 = self.ax3.scatter([0], [0], [0], c=[0], cmap=CMAP, s=0, alpha=0)
        self.st3 = self.ax3.scatter([], [], [], c=COLOR_STAR, marker='*', s=150, zorder=100)
        self.ax3.set_title("3. 3D View (Equal Scale)")

        # Sliders
        ax_min = self.fig.add_axes([0.15, 0.05, 0.3, 0.03])
        ax_max = self.fig.add_axes([0.55, 0.05, 0.3, 0.03])
        ax_ang = self.fig.add_axes([0.15, 0.09, 0.3, 0.03])

        self.s_min = Slider(ax_min, 'Min Z', 0, 100, valinit=0)
        self.s_max = Slider(ax_max, 'Max Z', 0, 100, valinit=100)
        # Angle 0-360
        self.s_ang = Slider(ax_ang, 'Angle', 0, 360, valinit=0)

        # 绑定事件
        self.s_min.on_changed(self.update_view)
        self.s_max.on_changed(self.update_view)
        self.s_ang.on_changed(self.update_view)

        # =========== 【补回丢失的定时器代码】 ===========
        # 如果没有这几行，auto_refresh_task 永远不会运行
        # AUTO_FIX_INTERVAL 需要确保在前面定义过 (比如 1000 或 2000 ms)
        self.timer = self.fig.canvas.new_timer(interval=10000)
        self.timer.add_callback(self.auto_refresh_task)
        self.timer.start()
        # ============================================

    def auto_refresh_task(self):
        if not self.collector.running:
            plt.close(self.fig)
            return

        # ================= 【关键修复】开始 =================
        # 原因：Collector 每次运行都会生成新的带时间戳的文件夹
        # Visualizer 必须实时获取这个新路径，否则它还在读旧文件或不存在的文件
        self.raw_csv = self.collector.csv_path
        self.fixed_csv = self.raw_csv.replace('.csv', '_fixed.csv')
        # ================= 【关键修复】结束 =================

        self.raw_csv = self.collector.csv_path
        self.fixed_csv = self.raw_csv.replace('.csv', '_fixed.csv')

        print(f"\n🔄 [Auto-Refresh] Checking new data...")

        # 这里的 run_fixer 建议也传参进去，或者确保 run_fixer 内部使用的是 self.raw_csv
        # 如果 run_fixer 是外部函数，可能需要修改调用方式；如果是类方法，确保它用的是更新后的 self.raw_csv
        self.run_fixer()

        if self.load_data_for_viz():
            # Update Slider Range
            self.s_min.valmin = self.z_min_global
            self.s_min.valmax = self.z_max_global
            self.s_max.valmin = self.z_min_global
            self.s_max.valmax = self.z_max_global

            # Update slider visual range
            self.s_min.ax.set_xlim(self.z_min_global, self.z_max_global)
            self.s_max.ax.set_xlim(self.z_min_global, self.z_max_global)

            # 强制更新滑块当前值 (为了防止滑块卡在旧数据的范围外，建议加个判断)
            # 如果当前值超出了新范围，才强制重置；否则保留用户手调的位置体验更好
            # 这里为了保险，先保留你的逻辑
            self.s_min.set_val(self.z_min_global)
            self.s_max.set_val(self.z_max_global)

            display_min = self.z_min_global
            display_max = self.z_max_global

            if display_max <= display_min:
                display_min -= 10.0  # 随便给个缓冲
                display_max += 10.0

            # 使用带缓冲的值来设置滑块背景的范围
            self.s_min.ax.set_xlim(display_min, display_max)
            self.s_max.ax.set_xlim(display_min, display_max)
            # =============================================================

            # 强制更新滑块当前值
            self.s_min.set_val(self.z_min_global)
            self.s_max.set_val(self.z_max_global)

            self.update_view(None)
            print(f"✅ [Auto-Refresh] Updated. Points: {len(self.df)}")
        else:
            # 调试用：打印一下它到底在读哪个文件，方便你确认修复是否生效
            print(f"⚠️ [Auto-Refresh] No data yet. (Target: {self.raw_csv})")

    def get_compass_string(self, angle):
        """辅助函数：将 0-360 转换为方位描述"""
        # 归一化到 0-360
        angle = angle % 360

        # 定义8个方位
        directions = ["北 (N)", "东北 (NE)", "东 (E)", "东南 (SE)",
                      "南 (S)", "西南 (SW)", "西 (W)", "西北 (NW)"]
        # 简单划分 (每45度一个扇区，中心偏移22.5度)
        index = int((angle + 22.5) // 45) % 8
        direction_str = directions[index]

        return f"{direction_str} {angle:.1f}°"

    def update_view(self, val=None):
        # 1. 卫语句：没数据就别干活
        if not hasattr(self, 'np_x') or self.np_x is None or len(self.np_x) == 0:
            return

        # 2. 获取 Slider 值
        z_min = self.s_min.val
        z_max = self.s_max.val
        angle_deg = self.s_ang.val
        angle_rad = np.radians(angle_deg)

        # 更新文字
        if hasattr(self, 'txt_angle_label'):
            self.txt_angle_label.set_text(self.get_compass_string(angle_deg))

        # 3. 数据准备 (交换 XY 以匹配上北右东)
        # 你的数据：X是纬度(北), Y是经度(东)，所以 plot_x 取 Y, plot_y 取 X
        plot_x = self.np_y
        plot_y = self.np_x
        plot_z = self.np_z

        # 4. 过滤 Z 轴范围 (加宽缓冲，确保能取到数据)
        # 打印调试：看看现在的筛选范围和数据范围
        # print(f"DEBUG: Z-Range[{z_min:.1f}, {z_max:.1f}] vs Data[{plot_z.min():.1f}, {plot_z.max():.1f}]")

        mask = (plot_z >= z_min - 100.0) & (plot_z <= z_max + 100.0)

        show_x = plot_x[mask]
        show_y = plot_y[mask]
        show_z = plot_z[mask]

        # 如果过滤完没点了，直接返回
        if len(show_x) == 0:
            return

        # 5. Profile 投影计算
        proj_dist = show_x * np.cos(angle_rad) + show_y * np.sin(angle_rad)

        # ================== 暴力重绘核心 ==================
        # 既然 set_offsets 有问题，我们直接 clear() 然后重新 scatter()
        # 这样 Matplotlib 会强制自动计算范围和颜色

        CMAP = 'viridis'  # 颜色映射

        # --- Ax1: Top View ---
        self.ax1.clear()  # 清除旧图
        self.ax1.set_title("1. Top View (XY)")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal', adjustable='datalim')  # 保持比例

        # 重新画散点，c=show_z 表示用高度决定颜色
        self.sc1 = self.ax1.scatter(show_x, show_y, c=show_z, cmap=CMAP, s=15, alpha=0.7)
        # 重新画主角位置 (红星) - 取最后一个点
        self.ax1.scatter(show_x[-1], show_y[-1], c='red', marker='*', s=150, zorder=100)

        # --- Ax2: Profile ---
        self.ax2.clear()
        self.ax2.set_title("2. Profile")
        self.ax2.grid(True, linestyle='--')
        # 侧视图不需要 equal aspect，否则 Z 轴会被压得很扁
        # self.ax2.set_aspect('equal')

        self.sc2 = self.ax2.scatter(proj_dist, show_z, c=show_z, cmap=CMAP, s=15, alpha=0.7)
        # 侧视图主角位置
        self.ax2.scatter(proj_dist[-1], show_z[-1], c='red', marker='*', s=150, zorder=100)

        # 强制设置 Y 轴范围匹配滑块 (但 X 轴让它自动缩放)
        self.ax2.set_ylim(z_min - 10, z_max + 10)

        # --- Ax3: 3D View ---
        self.ax3.clear()
        self.ax3.set_title("3. 3D View")

        # 3D 散点
        self.sc3 = self.ax3.scatter(show_x, show_y, show_z, c=show_z, cmap=CMAP, s=15, alpha=0.7)
        # 3D 主角
        self.ax3.scatter([show_x[-1]], [show_y[-1]], [show_z[-1]], c='red', marker='*', s=150, zorder=100)

        # 计算 3D 等比例 Box (这是 3D 视图不变形的关键)
        range_x = show_x.max() - show_x.min()
        range_y = show_y.max() - show_y.min()
        range_z = show_z.max() - show_z.min()
        max_range = np.array([range_x, range_y, range_z]).max() / 2.0

        if max_range < 1.0: max_range = 50.0

        mid_x = (show_x.max() + show_x.min()) * 0.5
        mid_y = (show_y.max() + show_y.min()) * 0.5
        mid_z = (show_z.max() + show_z.min()) * 0.5

        self.ax3.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax3.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax3.set_zlim(mid_z - max_range, mid_z + max_range)
        self.ax3.set_box_aspect((1, 1, 1))

        # 刷新画布
        self.fig.canvas.draw_idle()

    def on_pick(self, event):
        try:
            ind = event.ind[0]
            if hasattr(event.artist, '_offsets3d'):
                ox, oy, oz = event.artist._offsets3d
                px, py = ox[ind], oy[ind]
            else:
                offsets = event.artist.get_offsets()
                px, py = offsets[ind]

            dist = (self.df['X'] - px) ** 2 + (self.df['Y'] - py) ** 2
            closest_idx = dist.idxmin()

            if dist[closest_idx] < 50.0:
                row = self.df.loc[closest_idx]
                print(f"\n📍 Selected: [T:{row['Timestamp']}] Coords:({row['X']:.0f}, {row['Y']:.0f}, {row['Z']:.0f})")
                img_name = row['Image_Filename']
                if pd.notna(img_name) and str(img_name).strip() != "":
                    img_path = os.path.join(self.snapshots_dir, str(img_name))
                    if os.path.exists(img_path):
                        print(f"   📸 Opening: {img_name}")
                        os.startfile(img_path)
        except Exception as e:
            pass

    def show(self):
        plt.show()


# ==========================================
# MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    collector = MapDataCollector(base_output_dir="./game_data_output")
    collector.start()

    print("--- Visualizer Starting ---")
    time.sleep(2)

    viz = EngineeringVisualizer(collector)

    if viz.run_fixer() and viz.load_data_for_viz():
        viz.update_view(None)

    try:
        viz.show()
    except KeyboardInterrupt:
        pass

    print("Stopping collector...")
    collector.running = False
    collector.join()
    print("👋 Exited.")