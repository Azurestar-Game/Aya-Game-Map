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
# 🛠️ 【修复】同时导入 Line3DCollection 和 Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import os
from abc import ABC, abstractmethod

# ==========================================
# 🛠️ 字体与颜色配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 高对比度工程色谱
COLORS_LIST = ['#D73027', '#FC8D59', '#00CED1', '#4575B4', '#08306B']
CMAP_HEIGHT = LinearSegmentedColormap.from_list("Engineering", COLORS_LIST)

COLOR_PATH = '#AAAAAA'
COLOR_SPECIAL = '#FF00FF'


# ==========================================
# 0. 抽象基类 (数据加载)
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

            self.np_x = self.df['X'].values
            self.np_y = self.df['Y'].values
            self.np_z = self.df['Z'].values
            self.np_seg_id = self.df['segment_id'].values
            self.np_has_img = self.df['HasImage'].values.astype(bool)
            self.connect_mask = (self.np_seg_id[:-1] == self.np_seg_id[1:])

            # --- 计算全局边界 ---
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

            # 3D 中心与范围
            self.mid_x = (self.g_xmax + self.g_xmin) * 0.5
            self.mid_y = (self.g_ymax + self.g_ymin) * 0.5
            self.mid_z = (self.g_zmax + self.g_zmin) * 0.5

            rx, ry, rz = self.g_xmax - self.g_xmin, self.g_ymax - self.g_ymin, self.g_zmax - self.g_zmin
            self.max_range = max(rx, ry, rz) / 2.0
            if self.max_range < 100: self.max_range = 100

            print(f"✅ 数据加载成功: {len(self.df)} 点")
            return True
        except Exception as e:
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

            if dists[closest_idx] < 250000:
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
        print("🚀 启动可视化窗口...")
        plt.show(block=True)


# ==========================================
# 终极工程视图 (V3: 交互增强与辅助线版)
# ==========================================
class EngineeringVisualizer(MapVisualizer):
    def draw(self):
        # 1. 设置超大尺寸以满足像素需求
        self.fig = plt.figure(figsize=(20, 13))
        self.fig.canvas.manager.set_window_title(f"Engineering View V3 - {self.data_status}")

        self.fig.subplots_adjust(bottom=0.25, wspace=0.20, hspace=0.25)
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # 布局：左侧两图，右侧一图
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Top View
        self.ax2 = self.fig.add_subplot(2, 2, 3)  # Profile View
        self.ax3 = self.fig.add_subplot(1, 2, 2, projection='3d')  # 3D View

        # ----------------------------------------------------
        # 1. Top View (1000x1000px 等效, 可缩放移动)
        # ----------------------------------------------------
        self.lc1 = LineCollection([], colors=COLOR_PATH, linewidths=0.8, alpha=0.4)
        self.ax1.add_collection(self.lc1)
        self.sc1 = self.ax1.scatter([], [], c=[], cmap=CMAP_HEIGHT, s=20, alpha=0.9)
        self.st1 = self.ax1.scatter([], [], c=COLOR_SPECIAL, marker='*', s=200, edgecolors='black', zorder=100)

        self.ax1.set_title("1. Top View (North Up) - [可鼠标拖动缩放]")
        self.ax1.set_xlabel('East (Y)')
        self.ax1.set_ylabel('North (X)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')

        # 初始化锁定全局范围
        self.ax1.set_xlim(self.g_ymin, self.g_ymax)
        self.ax1.set_ylim(self.g_xmin, self.g_xmax)

        # ----------------------------------------------------
        # 2. Profile View (1000x500px 等效, 锁定范围, 辅助线)
        # ----------------------------------------------------
        self.lc2 = LineCollection([], colors=COLOR_PATH, linewidths=0.8, alpha=0.4)
        self.ax2.add_collection(self.lc2)
        self.sc2 = self.ax2.scatter([], [], c=[], cmap=CMAP_HEIGHT, s=20, alpha=0.9)
        self.st2 = self.ax2.scatter([], [], c=COLOR_SPECIAL, marker='*', s=200, edgecolors='black', zorder=100)

        # 辅助线
        self.line_min_z = self.ax2.axhline(self.g_zmin, color='cyan', linestyle='--', linewidth=1.5, label='Min Z')
        self.line_max_z = self.ax2.axhline(self.g_zmax, color='orange', linestyle='--', linewidth=1.5, label='Max Z')
        self.ax2.legend(loc='upper right')

        self.ax2.set_title("2. Profile View (Rotatable) - [Z轴范围锁定]")
        self.ax2.set_ylabel('Height Z')
        self.ax2.grid(True, linestyle='--')

        # ----------------------------------------------------
        # 3. 3D View (1500x1500px 等效, 切片平面)
        # ----------------------------------------------------
        dummy = np.array([[[0, 0, 0], [0, 0, 0]]])
        self.lc3 = Line3DCollection(dummy, colors=COLOR_PATH, linewidths=0.5, alpha=0.3)
        self.ax3.add_collection3d(self.lc3)
        self.sc3 = self.ax3.scatter([], [], [], c=[], cmap=CMAP_HEIGHT, s=20, alpha=0.9)
        self.st3 = self.ax3.scatter([], [], [], c=COLOR_SPECIAL, marker='*', s=200, edgecolors='black', zorder=100)

        # 占位平面
        self.plane_min = None
        self.plane_max = None

        self.ax3.set_title("3. 3D View")
        self.ax3.set_xlabel('East')
        self.ax3.set_ylabel('North')
        self.ax3.set_zlabel('Height')

        # 锁定 3D 盒子
        self.ax3.set_xlim(self.mid_y - self.max_range, self.mid_y + self.max_range)
        self.ax3.set_ylim(self.mid_x - self.max_range, self.mid_x + self.max_range)
        self.ax3.set_zlim(self.mid_z - self.max_range, self.mid_z + self.max_range)
        self.ax3.set_box_aspect((1, 1, 1))

        # --- 控件 ---
        ax_ang = self.fig.add_axes([0.15, 0.15, 0.65, 0.03])
        ax_max = self.fig.add_axes([0.15, 0.10, 0.65, 0.03])
        ax_min = self.fig.add_axes([0.15, 0.05, 0.65, 0.03])

        self.txt_ang = self.fig.text(0.15, 0.20, "Direction: N", fontsize=11, color='blue', fontweight='bold')

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

        # 2. 准备数据 (PlotX=East, PlotY=North)
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

        # --- Update Profile View ---
        self.sc2.set_offsets(np.c_[vis_proj, vis_z])
        self.sc2.set_array(vis_z)
        self.sc2.set_clim(self.g_zmin, self.g_zmax)

        if len(st_x) > 0:
            self.st2.set_offsets(np.c_[st_proj, st_z])
            self.st2.set_visible(True)
        else:
            self.st2.set_visible(False)

        p_min, p_max = proj_all.min(), proj_all.max()
        self.ax2.set_xlim(p_min - 100, p_max + 100)
        self.ax2.set_ylim(self.g_zmin - 100, self.g_zmax + 100)  # 强制锁定全局 Z

        # 更新辅助线
        self.line_min_z.set_ydata([z_min])
        self.line_max_z.set_ydata([z_max])

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

        # 绘制切片平面
        if self.plane_min: self.plane_min.remove()
        if self.plane_max: self.plane_max.remove()

        x_rng = [self.mid_y - self.max_range, self.mid_y + self.max_range]
        y_rng = [self.mid_x - self.max_range, self.mid_x + self.max_range]

        verts_min = [[(x_rng[0], y_rng[0], z_min), (x_rng[1], y_rng[0], z_min),
                      (x_rng[1], y_rng[1], z_min), (x_rng[0], y_rng[1], z_min)]]
        verts_max = [[(x_rng[0], y_rng[0], z_max), (x_rng[1], y_rng[0], z_max),
                      (x_rng[1], y_rng[1], z_max), (x_rng[0], y_rng[1], z_max)]]

        self.plane_min = Poly3DCollection(verts_min, alpha=0.2, facecolors='cyan')
        self.plane_max = Poly3DCollection(verts_max, alpha=0.2, facecolors='orange')

        self.ax3.add_collection3d(self.plane_min)
        self.ax3.add_collection3d(self.plane_max)

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
    TARGET = r"game_data_output/map_data_20251229_133655"
    viz = EngineeringVisualizer(TARGET)
    if viz.load_data():
        viz.draw()
        viz.show()