import matplotlib

# 强制使用 TkAgg 后端
try:
    matplotlib.use('TkAgg')
except:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import os
from abc import ABC, abstractmethod

# ==========================================
# 🛠️ 修复 1: 设置中文字体 (解决方块字问题)
# ==========================================
# 优先尝试 Windows 常见中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# --- 🎨 全局配置 (工程高对比版) ---
COLORS_LIST = [
    '#D73027',  # 低：深红
    '#FC8D59',  # 次低：橙红
    '#00CED1',  # 中：深绿松石 (Teal)
    '#4575B4',  # 高：皇家蓝
    '#08306B'  # 极高：深午夜蓝
]
CMAP_HEIGHT = LinearSegmentedColormap.from_list("Engineering", COLORS_LIST)

COLOR_PATH = '#AAAAAA'  # 连线颜色 (浅灰)
COLOR_SPECIAL = '#FF00FF'  # 特殊点 (洋红色)
COLOR_CURRENT = '#FFD700'  # 当前点 (金黄色)


# ==========================================
# 0. 抽象基类
# ==========================================
class MapVisualizer(ABC):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.snapshots_dir = os.path.join(folder_path, "map_snapshots")
        self.df = None
        self.fig = None

        fixed_path = os.path.join(folder_path, "map_matrix_fixed.csv")
        raw_path = os.path.join(folder_path, "map_matrix.csv")

        if os.path.exists(fixed_path):
            self.csv_path = fixed_path
            self.data_status = "✨ FIXED (修复版)"
        else:
            self.csv_path = raw_path
            self.data_status = "⚠️ RAW (原始版)"

    def load_data(self):
        if not os.path.exists(self.csv_path):
            print(f"❌ 找不到文件: {self.csv_path}")
            return False

        print(f"📂 正在加载数据: {self.data_status}")
        print(f"   路径: {self.csv_path}")

        try:
            raw_df = pd.read_csv(self.csv_path)
            is_valid = raw_df['X'].notna() & raw_df['Y'].notna() & raw_df['Z'].notna()
            raw_df['segment_id'] = (~is_valid).astype(int).cumsum()

            self.df = raw_df[is_valid].copy()
            for col in ['X', 'Y', 'Z']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            if 'Image_Filename' in self.df.columns:
                self.df['HasImage'] = self.df['Image_Filename'].fillna("").astype(str).str.strip() != ""
            else:
                self.df['HasImage'] = False

            # NumPy 转换
            self.np_x = self.df['X'].values
            self.np_y = self.df['Y'].values
            self.np_z = self.df['Z'].values
            self.np_seg_id = self.df['segment_id'].values
            self.np_has_img = self.df['HasImage'].values.astype(bool)
            self.connect_mask = (self.np_seg_id[:-1] == self.np_seg_id[1:])

            # --- 计算全局边界 (Global Bounds) ---
            self.g_xmin, self.g_xmax = self.np_x.min(), self.np_x.max()
            self.g_ymin, self.g_ymax = self.np_y.min(), self.np_y.max()
            self.g_zmin, self.g_zmax = self.np_z.min(), self.np_z.max()

            # 缓冲
            pad = 50
            self.g_xmin -= pad;
            self.g_xmax += pad
            self.g_ymin -= pad;
            self.g_ymax += pad
            self.g_zmin -= pad;
            self.g_zmax += pad

            # 3D 视图的全局中心和范围
            self.mid_x = (self.g_xmax + self.g_xmin) * 0.5
            self.mid_y = (self.g_ymax + self.g_ymin) * 0.5
            self.mid_z = (self.g_zmax + self.g_zmin) * 0.5

            rx = self.g_xmax - self.g_xmin
            ry = self.g_ymax - self.g_ymin
            rz = self.g_zmax - self.g_zmin
            self.max_range = max(rx, ry, rz) / 2.0
            if self.max_range < 100: self.max_range = 100

            print(f"✅ 数据加载成功: {len(self.df)} 点")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ 数据解析失败: {e}")
            return False

    def on_canvas_click(self, event):
        if not event.dblclick or event.button != 1: return
        if event.inaxes != self.ax1: return

        try:
            click_x, click_y = event.xdata, event.ydata
            df_imgs = self.df[self.df['HasImage']].copy()
            if df_imgs.empty: return

            dists = (df_imgs['Y'] - click_x) ** 2 + (df_imgs['X'] - click_y) ** 2
            closest_idx = dists.idxmin()

            if dists[closest_idx] < 250000:  # 500^2
                row = df_imgs.loc[closest_idx]
                img_name = row['Image_Filename']
                print(f"\n🎯 Double-Click: Opening {img_name} ...")
                img_path = os.path.join(self.snapshots_dir, str(img_name))
                if os.path.exists(img_path):
                    os.startfile(img_path)
                else:
                    print(f"❌ File not found: {img_path}")
        except:
            pass

    @abstractmethod
    def draw(self):
        pass

    def show(self):
        # 🛠️ 修复 3: 强制阻塞，防止窗口一闪而过
        print("🚀 启动可视化窗口...")
        plt.show(block=True)


# ==========================================
# 终极工程视图 (Stable Scaling 版)
# ==========================================
class EngineeringVisualizer(MapVisualizer):
    def draw(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title(f"Engineering View - {self.data_status}")

        self.fig.subplots_adjust(bottom=0.30, wspace=0.25, hspace=0.3)
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Top View
        self.ax2 = self.fig.add_subplot(2, 2, 3)  # Side View
        self.ax3 = self.fig.add_subplot(1, 2, 2, projection='3d')  # 3D View

        # --- 初始化图形对象 ---

        # 散点参数
        kw_sc = {'cmap': CMAP_HEIGHT, 's': 20, 'alpha': 0.9}
        kw_st = {'c': COLOR_SPECIAL, 'marker': '*', 's': 200, 'edgecolors': 'black', 'linewidths': 1.0, 'zorder': 100}

        # [1. Top View]
        self.lc1 = LineCollection([], colors=COLOR_PATH, linewidths=0.8, alpha=0.4)
        self.ax1.add_collection(self.lc1)
        # 🛠️ 修复 2: 显式传入 c=[] 解决 UserWarning
        self.sc1 = self.ax1.scatter([], [], c=[], **kw_sc)
        self.st1 = self.ax1.scatter([], [], **kw_st)

        self.ax1.set_title("1. Top View (North Up)")
        self.ax1.set_xlabel('East (Y)')
        self.ax1.set_ylabel('North (X)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')

        self.ax1.set_xlim(self.g_ymin, self.g_ymax)
        self.ax1.set_ylim(self.g_xmin, self.g_xmax)

        # [2. Side View]
        self.lc2 = LineCollection([], colors=COLOR_PATH, linewidths=0.8, alpha=0.4)
        self.ax2.add_collection(self.lc2)
        # 🛠️ 修复 2: 显式传入 c=[]
        self.sc2 = self.ax2.scatter([], [], c=[], **kw_sc)
        self.st2 = self.ax2.scatter([], [], **kw_st)

        self.ax2.set_title("2. Profile View (Rotatable)")
        self.ax2.set_ylabel('Height Z')
        self.ax2.grid(True, linestyle='--')

        # [3. 3D View]
        dummy = np.array([[[0, 0, 0], [0, 0, 0]]])
        self.lc3 = Line3DCollection(dummy, colors=COLOR_PATH, linewidths=0.5, alpha=0.3)
        self.ax3.add_collection3d(self.lc3)
        # 🛠️ 修复 2: 显式传入 c=[]
        self.sc3 = self.ax3.scatter([], [], [], c=[], **kw_sc)
        self.st3 = self.ax3.scatter([], [], [], **kw_st)

        self.ax3.set_title("3. 3D View")
        self.ax3.set_xlabel('East')
        self.ax3.set_ylabel('North')
        self.ax3.set_zlabel('Height')

        self.ax3.set_xlim(self.mid_y - self.max_range, self.mid_y + self.max_range)
        self.ax3.set_ylim(self.mid_x - self.max_range, self.mid_x + self.max_range)
        self.ax3.set_zlim(self.mid_z - self.max_range, self.mid_z + self.max_range)
        self.ax3.set_box_aspect((1, 1, 1))

        # --- 控件 ---
        ax_ang = self.fig.add_axes([0.15, 0.19, 0.65, 0.03])
        ax_max = self.fig.add_axes([0.15, 0.12, 0.65, 0.03])
        ax_min = self.fig.add_axes([0.15, 0.05, 0.65, 0.03])

        self.txt_ang = self.fig.text(0.15, 0.26, "Direction: N", fontsize=11, color='blue', fontweight='bold')

        self.s_min = Slider(ax_min, 'Min Z ', self.g_zmin, self.g_zmax, valinit=self.g_zmin, valfmt='%d')
        self.s_max = Slider(ax_max, 'Max Z ', self.g_zmin, self.g_zmax, valinit=self.g_zmax, valfmt='%d')
        self.s_ang = Slider(ax_ang, 'Angle ', 0, 360, valinit=0, valfmt='%.1f°')

        self.s_min.on_changed(self.update)
        self.s_max.on_changed(self.update)
        self.s_ang.on_changed(self.update)

        self.update(None)

    def get_compass(self, ang):
        dirs = ["北 (N)", "东北 (NE)", "东 (E)", "东南 (SE)", "南 (S)", "西南 (SW)", "西 (W)", "西北 (NW)"]
        idx = int((ang + 22.5) // 45) % 8
        return f"{dirs[idx]} {ang:.1f}°"

    def update(self, val):
        z_min, z_max = self.s_min.val, self.s_max.val
        if z_min > z_max: z_min = z_max

        ang = self.s_ang.val
        rad = np.radians(ang)
        self.txt_ang.set_text(self.get_compass(ang))

        # 1. 过滤可见性
        mask_p = (self.np_z >= z_min) & (self.np_z <= z_max)
        mask_l = mask_p[:-1] & mask_p[1:] & self.connect_mask

        # 2. 准备数据
        d_x, d_y, d_z = self.np_x, self.np_y, self.np_z

        # 3. 侧视图投影
        proj_all = d_y * np.cos(rad) + d_x * np.sin(rad)

        vis_x = d_y[mask_p]
        vis_y = d_x[mask_p]
        vis_z = d_z[mask_p]
        vis_proj = proj_all[mask_p]

        mask_st = mask_p & self.np_has_img
        st_x = d_y[mask_st]
        st_y = d_x[mask_st]
        st_z = d_z[mask_st]
        st_proj = proj_all[mask_st]

        # --- Update Top View ---
        self.sc1.set_offsets(np.c_[vis_x, vis_y])
        self.sc1.set_array(vis_z)
        self.sc1.set_clim(self.g_zmin, self.g_zmax)

        if len(st_x) > 0:
            self.st1.set_offsets(np.c_[st_x, st_y])
            self.st1.set_visible(True)
        else:
            self.st1.set_visible(False)

        if np.any(mask_l):
            l1 = np.stack([np.column_stack([d_y[:-1][mask_l], d_y[1:][mask_l]]),
                           np.column_stack([d_x[:-1][mask_l], d_x[1:][mask_l]])], axis=2)
            self.lc1.set_segments(l1)
        else:
            self.lc1.set_segments(np.zeros((0, 2, 2)))

        # --- Update Side View ---
        self.sc2.set_offsets(np.c_[vis_proj, vis_z])
        self.sc2.set_array(vis_z)
        self.sc2.set_clim(self.g_zmin, self.g_zmax)

        if len(st_x) > 0:
            self.st2.set_offsets(np.c_[st_proj, st_z])
            self.st2.set_visible(True)
        else:
            self.st2.set_visible(False)

        # 侧视图 X 轴范围基于全局投影
        p_min, p_max = proj_all.min(), proj_all.max()
        self.ax2.set_xlim(p_min - 100, p_max + 100)
        self.ax2.set_ylim(z_min - 100, z_max + 100)

        if np.any(mask_l):
            l2 = np.stack([np.column_stack([proj_all[:-1][mask_l], proj_all[1:][mask_l]]),
                           np.column_stack([d_z[:-1][mask_l], d_z[1:][mask_l]])], axis=2)
            self.lc2.set_segments(l2)
        else:
            self.lc2.set_segments(np.zeros((0, 2, 2)))

        # --- Update 3D View ---
        self.sc3._offsets3d = (vis_x, vis_y, vis_z)
        self.sc3.set_array(vis_z)
        self.sc3.set_clim(self.g_zmin, self.g_zmax)

        if len(st_x) > 0:
            self.st3._offsets3d = (st_x, st_y, st_z)
            self.st3.set_visible(True)
        else:
            self.st3.set_visible(False)

        if np.any(mask_l):
            l3 = np.stack([np.column_stack([d_y[:-1][mask_l], d_y[1:][mask_l]]),
                           np.column_stack([d_x[:-1][mask_l], d_x[1:][mask_l]]),
                           np.column_stack([d_z[:-1][mask_l], d_z[1:][mask_l]])], axis=2)
            self.lc3.set_segments(l3)
        else:
            self.lc3.set_segments(np.zeros((0, 2, 3)))

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # ⚠️ 请确认这里的路径是正确的
    TARGET = r"game_data_output/map_data_20251223_191201"
    viz = EngineeringVisualizer(TARGET)
    if viz.load_data():
        viz.draw()
        viz.show()