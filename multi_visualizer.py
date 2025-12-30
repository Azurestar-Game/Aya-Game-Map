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
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import os
from abc import ABC, abstractmethod

# ==========================================
# 🛠️ 字体与颜色配置 (完全保留)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 高对比度工程色谱
COLORS_LIST = ['#D73027', '#FC8D59', '#00CED1', '#4575B4', '#08306B']
CMAP_HEIGHT = LinearSegmentedColormap.from_list("Engineering", COLORS_LIST)

COLOR_PATH = '#AAAAAA'
COLOR_SPECIAL = '#FF00FF'


# ==========================================
# 0. 抽象基类 (已升级支持多 Target)
# ==========================================
class MapVisualizer(ABC):
    def __init__(self, target_list):
        """
        :param target_list: 包含多个文件夹路径的列表 ['path1', 'path2', ...]
        """
        self.target_list = target_list
        self.df = None
        self.fig = None

        # 这些变量现在是合并后的全局范围
        self.g_xmin, self.g_xmax = 0, 0
        self.g_ymin, self.g_ymax = 0, 0
        self.g_zmin, self.g_zmax = 0, 0
        self.mid_x, self.mid_y, self.mid_z = 0, 0, 0
        self.max_range = 100

    def load_data(self):
        all_dfs = []
        global_segment_offset = 0  # 用于防止不同文件的线段连在一起

        print(f"📦 准备加载 {len(self.target_list)} 个目标...")

        for folder_path in self.target_list:
            if not os.path.exists(folder_path):
                print(f"   ⚠️ 跳过无效路径: {folder_path}")
                continue

            # 1. 确定文件
            fixed_path = os.path.join(folder_path, "map_matrix_fixed.csv")
            raw_path = os.path.join(folder_path, "map_matrix.csv")
            csv_path = fixed_path if os.path.exists(fixed_path) else raw_path

            # 图片目录
            snapshots_dir = os.path.join(folder_path, "map_snapshots")

            if not os.path.exists(csv_path):
                print(f"   ⚠️ 找不到 CSV: {folder_path}")
                continue

            try:
                # 2. 读取单个文件
                sub_df = pd.read_csv(csv_path)

                # 清洗坐标
                is_valid = sub_df['X'].notna() & sub_df['Y'].notna() & sub_df['Z'].notna()
                sub_df = sub_df[is_valid].copy()

                # 3. 处理 Segment ID (线段断点)
                # 原始的 segment_id 是从 0 开始的，我们需要加上偏移量
                # 并在前面加一个 NaN 的“断层”以确保不和上一个文件相连（虽然 numpy mask 也能处理，但为了保险）
                sub_df['segment_id'] = sub_df.get('segment_id', 0)  # 如果没有就默认为0
                # 这里假设原始数据里 segment_id 已经是断开 NaN 后的累加值
                # 如果原始数据里没有 segment_id，下面的逻辑也能跑，只是所有点连成一条线

                # 如果没有 segment_id 列，我们手动根据索引创建一个简单的（假设连续）
                if 'segment_id' not in sub_df.columns:
                    sub_df['segment_id'] = 0

                # 加上全局偏移，保证不同文件的线段 id 不同
                sub_df['segment_id'] += global_segment_offset

                # 4. 处理图片路径 (计算绝对路径)
                if 'Image_Filename' in sub_df.columns:
                    sub_df['HasImage'] = sub_df['Image_Filename'].fillna("").astype(str).str.strip() != ""
                    # 创建一个新的列，存储图片的【绝对路径】
                    sub_df['Abs_Img_Path'] = sub_df.apply(
                        lambda row: os.path.join(snapshots_dir, str(row['Image_Filename'])) if row[
                            'HasImage'] else None,
                        axis=1
                    )
                else:
                    sub_df['HasImage'] = False
                    sub_df['Abs_Img_Path'] = None

                # 5. 更新偏移量
                if not sub_df.empty:
                    max_seg = sub_df['segment_id'].max()
                    global_segment_offset = max_seg + 10  # 加一点余量，确保断开

                all_dfs.append(sub_df)
                print(f"   ✅ 已合并: {os.path.basename(folder_path)} ({len(sub_df)} 点)")

            except Exception as e:
                print(f"   ❌ 解析失败 {folder_path}: {e}")

        # 6. 合并所有数据
        if not all_dfs:
            print("❌ 没有加载到任何有效数据。")
            return False

        self.df = pd.concat(all_dfs, ignore_index=True)

        # 转换数值类型
        for col in ['X', 'Y', 'Z']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 提取 Numpy 数组用于绘图
        self.np_x = self.df['X'].values
        self.np_y = self.df['Y'].values
        self.np_z = self.df['Z'].values
        self.np_seg_id = self.df['segment_id'].values
        self.np_has_img = self.df['HasImage'].values.astype(bool)

        # 【关键】重新计算连接掩码
        # 只要 segment_id 变了，这两个点之间就不能连线
        self.connect_mask = (self.np_seg_id[:-1] == self.np_seg_id[1:])

        # --- 计算全局边界 ---
        self.g_xmin, self.g_xmax = self.np_x.min(), self.np_x.max()
        self.g_ymin, self.g_ymax = self.np_y.min(), self.np_y.max()
        self.g_zmin, self.g_zmax = self.np_z.min(), self.np_z.max()

        pad = 50
        self.g_xmin -= pad;
        self.g_xmax += pad
        self.g_ymin -= pad;
        self.g_ymax += pad
        self.g_zmin -= pad;
        self.g_zmax += pad

        self.mid_x = (self.g_xmax + self.g_xmin) * 0.5
        self.mid_y = (self.g_ymax + self.g_ymin) * 0.5
        self.mid_z = (self.g_zmax + self.g_zmin) * 0.5

        rx, ry, rz = self.g_xmax - self.g_xmin, self.g_ymax - self.g_ymin, self.g_zmax - self.g_zmin
        self.max_range = max(rx, ry, rz) / 2.0
        if self.max_range < 100: self.max_range = 100

        print(f"🎉 全部数据合并完成: 总计 {len(self.df)} 点")
        return True

    def on_canvas_click(self, event):
        """双击查看图片 (支持多文件夹)"""
        if not event.dblclick or event.button != 1: return
        if event.inaxes != self.ax1: return

        try:
            click_x, click_y = event.xdata, event.ydata
            # 只筛选有图片的点
            df_imgs = self.df[self.df['HasImage']].copy()
            if df_imgs.empty: return

            # 计算距离
            dists = (df_imgs['Y'] - click_x) ** 2 + (df_imgs['X'] - click_y) ** 2
            closest_idx = dists.idxmin()

            # 阈值判定
            if dists[closest_idx] < 250000:  # 500*500 unit tolerance
                row = df_imgs.loc[closest_idx]
                img_path = row['Abs_Img_Path']  # 直接读取预先存好的绝对路径

                if img_path and os.path.exists(img_path):
                    print(f"\n🎯 Double-Click: Opening {os.path.basename(img_path)} ...")
                    os.startfile(img_path)
                else:
                    print(f"❌ File not found or path invalid: {img_path}")
        except Exception as e:
            print(f"Error handling click: {e}")

    @abstractmethod
    def draw(self):
        pass

    def show(self):
        print("🚀 启动可视化窗口 (Multi-Target)...")
        plt.show(block=True)


# ==========================================
# 终极工程视图 (V3: 交互增强与辅助线版)
# (这部分代码几乎不需要变动，除了 Title)
# ==========================================
class EngineeringVisualizer(MapVisualizer):
    def draw(self):
        # 1. 设置超大尺寸
        self.fig = plt.figure(figsize=(20, 13))
        # 显示加载了多少个 Targets
        title_str = f"Engineering View V3 - Merged {len(self.target_list)} Targets"
        self.fig.canvas.manager.set_window_title(title_str)

        self.fig.subplots_adjust(bottom=0.25, wspace=0.20, hspace=0.25)
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # 布局
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Top View
        self.ax2 = self.fig.add_subplot(2, 2, 3)  # Profile View
        self.ax3 = self.fig.add_subplot(1, 2, 2, projection='3d')  # 3D View

        # ----------------------------------------------------
        # 1. Top View
        # ----------------------------------------------------
        self.lc1 = LineCollection([], colors=COLOR_PATH, linewidths=0.8, alpha=0.4)
        self.ax1.add_collection(self.lc1)
        self.sc1 = self.ax1.scatter([], [], c=[], cmap=CMAP_HEIGHT, s=20, alpha=0.9)
        self.st1 = self.ax1.scatter([], [], c=COLOR_SPECIAL, marker='*', s=200, edgecolors='black', zorder=100)

        self.ax1.set_title("1. Top View (North Up)")
        self.ax1.set_xlabel('East (Y)')
        self.ax1.set_ylabel('North (X)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlim(self.g_ymin, self.g_ymax)
        self.ax1.set_ylim(self.g_xmin, self.g_xmax)

        # ----------------------------------------------------
        # 2. Profile View
        # ----------------------------------------------------
        self.lc2 = LineCollection([], colors=COLOR_PATH, linewidths=0.8, alpha=0.4)
        self.ax2.add_collection(self.lc2)
        self.sc2 = self.ax2.scatter([], [], c=[], cmap=CMAP_HEIGHT, s=20, alpha=0.9)
        self.st2 = self.ax2.scatter([], [], c=COLOR_SPECIAL, marker='*', s=200, edgecolors='black', zorder=100)

        self.line_min_z = self.ax2.axhline(self.g_zmin, color='cyan', linestyle='--', linewidth=1.5)
        self.line_max_z = self.ax2.axhline(self.g_zmax, color='orange', linestyle='--', linewidth=1.5)

        self.ax2.set_title("2. Profile View (Rotatable)")
        self.ax2.set_ylabel('Height Z')
        self.ax2.grid(True, linestyle='--')

        # ----------------------------------------------------
        # 3. 3D View
        # ----------------------------------------------------
        dummy = np.array([[[0, 0, 0], [0, 0, 0]]])
        self.lc3 = Line3DCollection(dummy, colors=COLOR_PATH, linewidths=0.5, alpha=0.3)
        self.ax3.add_collection3d(self.lc3)
        self.sc3 = self.ax3.scatter([], [], [], c=[], cmap=CMAP_HEIGHT, s=20, alpha=0.9)
        self.st3 = self.ax3.scatter([], [], [], c=COLOR_SPECIAL, marker='*', s=200, edgecolors='black', zorder=100)

        self.plane_min = None
        self.plane_max = None

        self.ax3.set_title("3. 3D View")
        self.ax3.set_xlabel('East')
        self.ax3.set_ylabel('North')
        self.ax3.set_zlabel('Height')
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

        # 1. 过滤可见性 (Data Processing)
        mask_p = (self.np_z >= z_min) & (self.np_z <= z_max)
        # 【关键】这里用到了 self.connect_mask，它已经包含了跨文件断开的逻辑
        mask_l = mask_p[:-1] & mask_p[1:] & self.connect_mask

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
        self.ax2.set_ylim(self.g_zmin - 100, self.g_zmax + 100)

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
    # ==========================================
    # 🛠️ 在这里配置你的 Target 列表
    # ==========================================
    TARGETS = [
        r"game_data_output/map_data_20251224_140944",
        r"game_data_output/map_data_20251224_152637",
        r"game_data_output/map_data_20251224_163308",
        r"game_data_output/map_data_20251224_185559",
        r"game_data_output/map_data_20251224_200336",
        r"game_data_output/map_data_20251224_211137",
        r"game_data_output/map_data_20251224_213933"
        # 你可以继续添加 target3, target4 ...
    ]

    # 只需要把列表传进去
    viz = EngineeringVisualizer(TARGETS)
    if viz.load_data():
        viz.draw()
        viz.show()