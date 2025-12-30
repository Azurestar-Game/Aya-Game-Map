import matplotlib

# 强制使用 TkAgg 后端以确保交互窗口正常弹出
try:
    matplotlib.use('TkAgg')
except:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from matplotlib.colors import LinearSegmentedColormap
import os
from abc import ABC, abstractmethod

# ==========================================
# 🛠️ 配置区域
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置
COLORS_LIST = ['#D73027', '#FC8D59', '#fee090', '#e0f3f8', '#91bfdb', '#4575B4']
CMAP_HEIGHT = LinearSegmentedColormap.from_list("Engineering", COLORS_LIST)
COLOR_PATH = '#AAAAAA'
COLOR_SPECIAL = '#FF00FF'  # 图片点颜色


# ==========================================
# 1. 核心逻辑：智能分层算法
# ==========================================
def get_layer_name(x, y, z, source_folder):
    """
    根据坐标和源文件，判定该点属于哪一层。
    依据：用户提供的基准坐标和重叠区逻辑。
    """
    # 0. 优先判定商业区 (如果需要单独看商业区，取消下面注释)
    # if "商业区" in source_folder or "213933" in source_folder:
    #     return "Commercial"
    # 注：根据需求，商业区也也有 Level 1-5，建议按高度混入各层，或者单独通过 UI 过滤

    # 1. Monster / Level F (最底层)
    # 基准 -11624
    if z < -10800:
        return "Monster"

    # 2. Level 5
    # 基准 -10000, 常见 -10000~-10346
    # Level 4 的 -10000 部分通往 L5，视为 L5
    if z < -9200:
        return "Level 5"

    # 3. Level 4
    # 基准 -8000, 常见 -7900, -8500~-9500
    if z < -6000:
        return "Level 4"

    # 4. Level 3
    # 基准 -5250, 常见 -4700, -5100
    if z < -4200:
        return "Level 3"

    # 5. Level 1 vs Level 2 (复杂重叠区)
    # Level 2 基准 -2750. 常见 -1800, -2200, -2800
    # Level 1 基准 -750.  常见 -700, -1300, -1900
    # 重叠区: Z 在 -1500 到 -2500 之间
    if z < -1500:
        # 进入 L1/L2 纠缠区
        if z > -2500:
            # === 核心逻辑：利用 Y 轴分离 ===
            # L1 (-1900) Y range: 8000 to 17000 (Higher Y)
            # L2 (-1800) Y range: -5000 to 5000 (Lower Y)
            # 分界线取 Y = 6500 (介于 5000 和 8000 之间)
            if y < 6500:
                return "Level 2"
            else:
                return "Level 1"
        else:
            # 明显是 Level 2 (e.g. -2750)
            return "Level 2"

    # 6. Entrance / Level 1 Main
    # Level 1 上界接近 0? Entrance 是 0
    if z > -200:
        return "Entrance"

    return "Level 1"


# ==========================================
# 2. 数据加载类
# ==========================================
class MapVisualizer:
    def __init__(self, target_list):
        self.target_list = target_list
        self.df = None
        self.fig = None

        # 默认视图范围
        self.view_presets = {
            "All": (-12000, 1000),
            "Entrance": (-200, 500),
            "Level 1": (-2000, 0),
            "Level 2": (-4200, -1500),
            "Level 3": (-5500, -4200),
            "Level 4": (-9200, -6000),
            "Level 5": (-10800, -9200),
            "Monster": (-13000, -10800),
            "Commercial": (-11000, 0)  # 商业区跨度很大
        }

    def load_data(self):
        all_dfs = []
        global_seg_offset = 0

        print(f"📦 正在处理 {len(self.target_list)} 个数据源...")

        for folder_path in self.target_list:
            if not os.path.exists(folder_path): continue

            # 确定文件名
            fixed = os.path.join(folder_path, "map_matrix_fixed.csv")
            raw = os.path.join(folder_path, "map_matrix.csv")
            fpath = fixed if os.path.exists(fixed) else raw

            if not os.path.exists(fpath): continue

            try:
                sub_df = pd.read_csv(fpath)
                # 清洗
                sub_df = sub_df[sub_df['X'].notna() & sub_df['Y'].notna() & sub_df['Z'].notna()].copy()

                # 线段ID偏移
                if 'segment_id' not in sub_df.columns: sub_df['segment_id'] = 0
                sub_df['segment_id'] += global_seg_offset

                # 图片路径
                snap_dir = os.path.join(folder_path, "map_snapshots")
                if 'Image_Filename' in sub_df.columns:
                    sub_df['HasImage'] = sub_df['Image_Filename'].fillna("").str.strip() != ""
                    sub_df['Abs_Img_Path'] = sub_df.apply(
                        lambda r: os.path.join(snap_dir, str(r['Image_Filename'])) if r['HasImage'] else None, axis=1
                    )
                else:
                    sub_df['HasImage'] = False
                    sub_df['Abs_Img_Path'] = None

                # === 关键：添加元数据 ===
                # 标记是否为商业区源文件
                is_commercial = "商业区" in folder_path or "213933" in folder_path
                sub_df['IsCommercial'] = is_commercial
                sub_df['Source'] = os.path.basename(folder_path)

                # === 关键：预计算楼层 ===
                # 这样绘图时过滤速度极快
                sub_df['Layer'] = sub_df.apply(
                    lambda r: get_layer_name(r['X'], r['Y'], r['Z'], r['Source']), axis=1
                )

                if not sub_df.empty:
                    global_seg_offset = sub_df['segment_id'].max() + 100

                all_dfs.append(sub_df)
                print(f"   ✅ Loaded: {sub_df['Source'].iloc[0] if not sub_df.empty else 'Empty'}")

            except Exception as e:
                print(f"   ❌ Error {folder_path}: {e}")

        if not all_dfs: return False

        self.df = pd.concat(all_dfs, ignore_index=True)
        for c in ['X', 'Y', 'Z']: self.df[c] = pd.to_numeric(self.df[c])

        # 转为 numpy 加速
        self.np_x = self.df['X'].values
        self.np_y = self.df['Y'].values
        self.np_z = self.df['Z'].values
        self.np_seg = self.df['segment_id'].values
        self.np_has_img = self.df['HasImage'].values
        self.np_layer = self.df['Layer'].values
        self.np_is_comm = self.df['IsCommercial'].values

        self.connect_mask = (self.np_seg[:-1] == self.np_seg[1:])

        # 全局范围
        pad = 100
        self.g_xmin, self.g_xmax = self.np_x.min() - pad, self.np_x.max() + pad
        self.g_ymin, self.g_ymax = self.np_y.min() - pad, self.np_y.max() + pad
        self.g_zmin, self.g_zmax = self.np_z.min() - pad, self.np_z.max() + pad

        self.mid_x = (self.g_xmax + self.g_xmin) / 2
        self.mid_y = (self.g_ymax + self.g_ymin) / 2
        self.mid_z = (self.g_zmax + self.g_zmin) / 2
        self.max_range = max(self.g_xmax - self.g_xmin, self.g_ymax - self.g_ymin, self.g_zmax - self.g_zmin) / 2

        return True

    def on_canvas_click(self, event):
        if not event.dblclick or event.button != 1: return
        if event.inaxes != self.ax1: return
        try:
            # 仅在当前可见的点中搜索 (优化性能)
            # 这里简化为全局搜索，因为数据量通常不大
            df_img = self.df[self.df['HasImage']]
            if df_img.empty: return

            dists = (df_img['Y'] - event.xdata) ** 2 + (df_img['X'] - event.ydata) ** 2
            closest = dists.idxmin()
            if dists[closest] < 250000:
                path = df_img.loc[closest, 'Abs_Img_Path']
                if path and os.path.exists(path):
                    print(f"🖼️ Opening: {path}")
                    os.startfile(path)
        except:
            pass


# ==========================================
# 3. 可视化界面 (含楼层选择器)
# ==========================================
class EngineeringVisualizer(MapVisualizer):
    def draw(self):
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.canvas.manager.set_window_title("Kappa Base Engineering View")

        # 布局定义
        # 左侧预留给控件，中间 Top/Profile，右侧 3D
        self.ax1 = self.fig.add_axes([0.15, 0.55, 0.35, 0.40])  # Top
        self.ax2 = self.fig.add_axes([0.15, 0.10, 0.35, 0.35])  # Profile
        self.ax3 = self.fig.add_axes([0.55, 0.10, 0.40, 0.85], projection='3d')  # 3D

        # 控件区 (左侧 0.0 ~ 0.12)
        ax_radio = self.fig.add_axes([0.01, 0.60, 0.10, 0.30], facecolor='#f0f0f0')
        ax_check = self.fig.add_axes([0.01, 0.50, 0.10, 0.08], facecolor='#f0f0f0')
        ax_slider_min = self.fig.add_axes([0.15, 0.05, 0.35, 0.02])
        ax_slider_max = self.fig.add_axes([0.15, 0.02, 0.35, 0.02])
        ax_slider_ang = self.fig.add_axes([0.02, 0.15, 0.10, 0.02])

        # === 初始化绘图对象 ===
        # Top View
        self.lc1 = LineCollection([], linewidths=0.8, alpha=0.5, colors=COLOR_PATH)
        self.sc1 = self.ax1.scatter([], [], s=15, cmap=CMAP_HEIGHT, alpha=0.8)
        self.st1 = self.ax1.scatter([], [], marker='*', s=150, c=COLOR_SPECIAL, edgecolors='k', zorder=100)
        self.ax1.add_collection(self.lc1)
        self.ax1.set_title("Top View (North Up)")
        self.ax1.set_xlabel("East (Y)");
        self.ax1.set_ylabel("North (X)")
        self.ax1.set_xlim(self.g_ymin, self.g_ymax)
        self.ax1.set_ylim(self.g_xmin, self.g_xmax)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)

        # Profile View
        self.lc2 = LineCollection([], linewidths=0.8, alpha=0.5, colors=COLOR_PATH)
        self.sc2 = self.ax2.scatter([], [], s=15, cmap=CMAP_HEIGHT, alpha=0.8)
        self.st2 = self.ax2.scatter([], [], marker='*', s=150, c=COLOR_SPECIAL, edgecolors='k', zorder=100)
        self.ax2.add_collection(self.lc2)
        self.ax2.set_title("Profile View (Rotatable)")
        self.ax2.set_ylabel("Height (Z)")
        self.ax2.grid(True, linestyle='--')
        self.line_zmin = self.ax2.axhline(0, c='cyan', ls='--')
        self.line_zmax = self.ax2.axhline(0, c='orange', ls='--')

        # 3D View
        # self.lc3 = Line3DCollection([], linewidths=0.5, alpha=0.3, colors=COLOR_PATH)

        # 3D View
        # 🛠️ 【修复】这里必须给一个占位数据，不能是空列表 []，否则 add_collection3d 会报错
        dummy_segments = np.array([[[0, 0, 0], [0, 0, 0]]])
        self.lc3 = Line3DCollection(dummy_segments, linewidths=0.5, alpha=0.3, colors=COLOR_PATH)

        # 下面这两行 scatter 如果报 Warning 可以忽略，或者也给个占位符，但不会导致崩溃
        self.sc3 = self.ax3.scatter([], [], [], s=10, cmap=CMAP_HEIGHT, alpha=0.8)
        self.st3 = self.ax3.scatter([], [], [], marker='*', s=150, c=COLOR_SPECIAL, edgecolors='k')

        self.ax3.add_collection3d(self.lc3)
        self.ax3.set_xlabel('East');
        self.ax3.set_ylabel('North');
        self.ax3.set_zlabel('Height')

        self.sc3 = self.ax3.scatter([], [], [], s=10, cmap=CMAP_HEIGHT, alpha=0.8)
        self.st3 = self.ax3.scatter([], [], [], marker='*', s=150, c=COLOR_SPECIAL, edgecolors='k')
        self.ax3.add_collection3d(self.lc3)
        self.ax3.set_xlabel('East');
        self.ax3.set_ylabel('North');
        self.ax3.set_zlabel('Height')
        self.ax3.set_xlim(self.mid_y - self.max_range, self.mid_y + self.max_range)
        self.ax3.set_ylim(self.mid_x - self.max_range, self.mid_x + self.max_range)
        self.ax3.set_zlim(self.mid_z - self.max_range, self.mid_z + self.max_range)

        # === 控件逻辑 ===
        self.cur_layer = "All"
        self.show_comm_only = False

        # 1. 楼层选择器
        layers = ["All", "Entrance", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Monster"]
        self.radio = RadioButtons(ax_radio, layers, active=0)
        self.radio.on_clicked(self.on_layer_change)

        # 2. 商业区过滤器
        self.check = CheckButtons(ax_check, ["Only Commercial"], [False])
        self.check.on_clicked(self.on_check_change)

        # 3. 滑动条
        self.s_min = Slider(ax_slider_min, 'Min Z', self.g_zmin, self.g_zmax, valinit=self.g_zmin)
        self.s_max = Slider(ax_slider_max, 'Max Z', self.g_zmin, self.g_zmax, valinit=self.g_zmax)
        self.s_ang = Slider(ax_slider_ang, 'Angle', 0, 360, valinit=0)

        self.s_min.on_changed(self.update_plot)
        self.s_max.on_changed(self.update_plot)
        self.s_ang.on_changed(self.update_plot)

        # 绑定点击
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # 初始化
        self.update_plot(None)

    def on_layer_change(self, label):
        self.cur_layer = label
        # 自动调整 Z 轴范围到该层的典型值
        z_range = self.view_presets.get(label, (self.g_zmin, self.g_zmax))
        # 临时静默更新 slider，防止触发两次重绘
        self.s_min.eventson = False
        self.s_max.eventson = False
        self.s_min.set_val(z_range[0])
        self.s_max.set_val(z_range[1])
        self.s_min.eventson = True
        self.s_max.eventson = True
        self.update_plot(None)

    def on_check_change(self, label):
        self.show_comm_only = not self.show_comm_only
        self.update_plot(None)

    def update_plot(self, val):
        z_min, z_max = self.s_min.val, self.s_max.val
        angle = np.radians(self.s_ang.val)

        # === 核心过滤逻辑 ===
        # 1. 基础 Z 轴过滤
        mask = (self.np_z >= z_min) & (self.np_z <= z_max)

        # 2. 楼层过滤 (Layer Filter)
        if self.cur_layer != "All":
            mask = mask & (self.np_layer == self.cur_layer)

        # 3. 商业区过滤
        if self.show_comm_only:
            mask = mask & (self.np_is_comm)

        # 4. 线段连通性 (必须同时满足可见性 + 原始连通性)
        mask_lines = mask[:-1] & mask[1:] & self.connect_mask

        # === 准备数据 ===
        vis_x = self.np_x[mask]
        vis_y = self.np_y[mask]
        vis_z = self.np_z[mask]

        # 投影计算
        proj = vis_y * np.cos(angle) + vis_x * np.sin(angle)

        # 图片点
        mask_st = mask & self.np_has_img
        st_x = self.np_x[mask_st]
        st_y = self.np_y[mask_st]
        st_z = self.np_z[mask_st]
        st_proj = st_y * np.cos(angle) + st_x * np.sin(angle)

        # === 绘图更新 ===

        # 1. Top View
        self.sc1.set_offsets(np.c_[vis_y, vis_x])  # Note: Y is East (X-axis on plot), X is North (Y-axis on plot)
        self.sc1.set_array(vis_z)
        self.sc1.set_clim(self.g_zmin, self.g_zmax)
        if len(st_x) > 0:
            self.st1.set_offsets(np.c_[st_y, st_x])
            self.st1.set_visible(True)
        else:
            self.st1.set_visible(False)

        if np.any(mask_lines):
            # segs shape: (N, 2, 2) -> (x1, y1) to (x2, y2)
            # Plot X axis is Data Y, Plot Y axis is Data X
            segs = np.stack([np.column_stack([self.np_y[:-1][mask_lines], self.np_y[1:][mask_lines]]),
                             np.column_stack([self.np_x[:-1][mask_lines], self.np_x[1:][mask_lines]])], axis=2)
            self.lc1.set_segments(segs)
        else:
            self.lc1.set_segments(np.zeros((0, 2, 2)))

        # 2. Profile View
        self.sc2.set_offsets(np.c_[proj, vis_z])
        self.sc2.set_array(vis_z)
        self.sc2.set_clim(self.g_zmin, self.g_zmax)
        if len(st_x) > 0:
            self.st2.set_offsets(np.c_[st_proj, st_z])
            self.st2.set_visible(True)
        else:
            self.st2.set_visible(False)

        # Profile View 自适应 X 轴
        if len(proj) > 0:
            p_min, p_max = proj.min(), proj.max()
            self.ax2.set_xlim(p_min - 100, p_max + 100)

        self.line_zmin.set_ydata([z_min])
        self.line_zmax.set_ydata([z_max])

        if np.any(mask_lines):
            proj_all = self.np_y * np.cos(angle) + self.np_x * np.sin(angle)
            segs_p = np.stack([np.column_stack([proj_all[:-1][mask_lines], proj_all[1:][mask_lines]]),
                               np.column_stack([self.np_z[:-1][mask_lines], self.np_z[1:][mask_lines]])], axis=2)
            self.lc2.set_segments(segs_p)
        else:
            self.lc2.set_segments(np.zeros((0, 2, 2)))

        # 3. 3D View
        self.sc3._offsets3d = (vis_y, vis_x, vis_z)  # Plot x=East(Y), y=North(X)
        self.sc3.set_array(vis_z)
        self.sc3.set_clim(self.g_zmin, self.g_zmax)
        if len(st_x) > 0:
            self.st3._offsets3d = (st_y, st_x, st_z)
            self.st3.set_visible(True)
        else:
            self.st3.set_visible(False)

        if np.any(mask_lines):
            segs_3 = np.stack([np.column_stack([self.np_y[:-1][mask_lines], self.np_y[1:][mask_lines]]),
                               np.column_stack([self.np_x[:-1][mask_lines], self.np_x[1:][mask_lines]]),
                               np.column_stack([self.np_z[:-1][mask_lines], self.np_z[1:][mask_lines]])], axis=2)
            self.lc3.set_segments(segs_3)
        else:
            self.lc3.set_segments(np.zeros((0, 2, 3)))

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ==========================================
# 4. 执行入口
# ==========================================
if __name__ == "__main__":
    TARGETS = [
        r"game_data_output/map_data_20251224_140944",  # 1F
        r"game_data_output/map_data_20251224_152637",  # Core Elevator
        r"game_data_output/map_data_20251224_163308",  # 2F
        r"game_data_output/map_data_20251224_185559",  # 3F
        r"game_data_output/map_data_20251224_200336",  # 4F
        r"game_data_output/map_data_20251224_211137",  # 5F
        r"game_data_output/map_data_20251224_213933",  # Commercial
    ]

    viz = EngineeringVisualizer(TARGETS)
    if viz.load_data():
        viz.draw()
        viz.show()