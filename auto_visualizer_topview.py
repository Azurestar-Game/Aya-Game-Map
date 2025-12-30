import matplotlib

# 强制使用 TkAgg 后端以支持交互
try:
    matplotlib.use('TkAgg')
except:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# 高对比度工程色谱 (用于表示高度)
COLORS_LIST = ['#D73027', '#FC8D59', '#00CED1', '#4575B4', '#08306B']
CMAP_HEIGHT = LinearSegmentedColormap.from_list("Engineering", COLORS_LIST)

COLOR_PATH = '#AAAAAA'
COLOR_SPECIAL = '#FF00FF'


# ==========================================
# 0. 抽象基类 (数据加载 - 保持不变)
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
            # 过滤无效坐标
            is_valid = raw_df['X'].notna() & raw_df['Y'].notna() & raw_df['Z'].notna()
            # 标记段ID用于断点
            raw_df['segment_id'] = (~is_valid).astype(int).cumsum()

            self.df = raw_df[is_valid].copy()
            for col in ['X', 'Y', 'Z']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            if 'Image_Filename' in self.df.columns:
                self.df['HasImage'] = self.df['Image_Filename'].fillna("").astype(str).str.strip() != ""
            else:
                self.df['HasImage'] = False

            # 转为 Numpy 数组加速
            self.np_x = self.df['X'].values
            self.np_y = self.df['Y'].values
            self.np_z = self.df['Z'].values
            self.np_seg_id = self.df['segment_id'].values
            self.np_has_img = self.df['HasImage'].values.astype(bool)
            # 连接掩码：同一段内且连续的点
            self.connect_mask = (self.np_seg_id[:-1] == self.np_seg_id[1:])

            # 计算边界
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

            print(f"✅ 数据加载成功: {len(self.df)} 点")
            return True
        except Exception as e:
            print(f"❌ 数据解析失败: {e}")
            return False

    def on_canvas_click(self, event):
        """处理鼠标双击：打开对应位置的截图"""
        if not event.dblclick or event.button != 1: return
        if event.inaxes != self.ax: return  # 仅在主绘图区响应

        try:
            click_x, click_y = event.xdata, event.ydata  # xdata是Y轴(东), ydata是X轴(北)
            df_imgs = self.df[self.df['HasImage']].copy()
            if df_imgs.empty: return

            # 计算距离 (注意坐标系: event.xdata 对应 Y, event.ydata 对应 X)
            dists = (df_imgs['Y'] - click_x) ** 2 + (df_imgs['X'] - click_y) ** 2
            closest_idx = dists.idxmin()

            # 点击容差 (单位: 游戏坐标距离平方)
            if dists[closest_idx] < 250000:  # 500*500
                row = df_imgs.loc[closest_idx]
                img_name = row['Image_Filename']
                print(f"\n🎯 Double-Click: Opening {img_name} ...")
                img_path = os.path.join(self.snapshots_dir, str(img_name))
                if os.path.exists(img_path):
                    os.startfile(img_path)  # Windows
                else:
                    print(f"❌ File not found: {img_path}")
        except Exception as e:
            print(f"交互错误: {e}")

    @abstractmethod
    def draw(self):
        pass

    def show(self):
        print("🚀 启动 Top View 可视化窗口...")
        plt.show(block=True)


# ==========================================
# 专注于 Top View 的可视化类
# ==========================================
class TopViewVisualizer(MapVisualizer):
    def draw(self):
        # 1. 窗口设置 (单一大图)
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.manager.set_window_title(f"Top View Only - {self.data_status}")

        # 预留底部空间给滑块
        self.fig.subplots_adjust(bottom=0.15)
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # ----------------------------------------------------
        # 绘图对象初始化
        # ----------------------------------------------------
        # 1. 路径线 (LineCollection)
        self.lc = LineCollection([], colors=COLOR_PATH, linewidths=1.0, alpha=0.5)
        self.ax.add_collection(self.lc)

        # 2. 路径点 (Scatter) - 颜色表示高度
        self.sc = self.ax.scatter([], [], c=[], cmap=CMAP_HEIGHT, s=15, alpha=0.9, label='Path Points')

        # 3. 带有图片的特殊点 (Star Marker)
        self.st = self.ax.scatter([], [], c=COLOR_SPECIAL, marker='*', s=180, edgecolors='black', zorder=100,
                                  label='Screenshot')

        # 坐标轴设置 (North Up: X轴向上, Y轴向右)
        self.ax.set_title("🗺️ Map Top View (North Up)\n[滚动:缩放] [右键:平移] [双击星号:看图]", fontsize=12)
        self.ax.set_xlabel('East (Y)')
        self.ax.set_ylabel('North (X)')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_aspect('equal')  # 保证比例不拉伸

        # 设置初始范围
        self.ax.set_xlim(self.g_ymin, self.g_ymax)
        self.ax.set_ylim(self.g_xmin, self.g_xmax)

        # 添加 Colorbar 显示高度图例
        cbar = plt.colorbar(self.sc, ax=self.ax, fraction=0.03, pad=0.04)
        cbar.set_label('Height (Z)')

        # ----------------------------------------------------
        # 控件 (仅保留 Z 轴范围过滤，方便看楼层)
        # ----------------------------------------------------
        ax_max = self.fig.add_axes([0.20, 0.06, 0.60, 0.03])
        ax_min = self.fig.add_axes([0.20, 0.02, 0.60, 0.03])

        self.s_min = Slider(ax_min, 'Min Height (Z) ', self.g_zmin, self.g_zmax, valinit=self.g_zmin, valfmt='%d')
        self.s_max = Slider(ax_max, 'Max Height (Z) ', self.g_zmin, self.g_zmax, valinit=self.g_zmax, valfmt='%d')

        self.s_min.on_changed(self.update)
        self.s_max.on_changed(self.update)

        # 初始化绘制
        self.update(None)

    def update(self, val):
        z_min, z_max = self.s_min.val, self.s_max.val
        if z_min > z_max: z_min = z_max  # 防止交叉

        # 1. 过滤可见性 (Height Slicing)
        mask_p = (self.np_z >= z_min) & (self.np_z <= z_max)
        # 连线要求：两点都可见，且属于同一段
        mask_l = mask_p[:-1] & mask_p[1:] & self.connect_mask

        # 2. 提取数据 (注意：Plot X轴对应数据的Y, Plot Y轴对应数据的X)
        d_x, d_y, d_z = self.np_x, self.np_y, self.np_z

        vis_x = d_y[mask_p]  # East
        vis_y = d_x[mask_p]  # North
        vis_z = d_z[mask_p]  # Height (Color)

        # 带有图片的点
        mask_st = mask_p & self.np_has_img
        st_x = d_y[mask_st]
        st_y = d_x[mask_st]

        # --- 更新散点 ---
        self.sc.set_offsets(np.c_[vis_x, vis_y])
        self.sc.set_array(vis_z)
        self.sc.set_clim(self.g_zmin, self.g_zmax)

        # --- 更新五角星 ---
        if len(st_x) > 0:
            self.st.set_offsets(np.c_[st_x, st_y])
            self.st.set_visible(True)
        else:
            self.st.set_visible(False)

        # --- 更新连线 ---
        if np.any(mask_l):
            # LineCollection 需要 segments 格式: (N, 2, 2) -> (x0, y0) to (x1, y1)
            # 同样注意：X coord is d_y, Y coord is d_x
            points = np.array([d_y, d_x]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            self.lc.set_segments(segments[mask_l])
        else:
            self.lc.set_segments(np.zeros((0, 2, 2)))

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # ⚠️ 请确认这里的路径是正确的
    TARGET = r"game_data_output/map_data_20251229_170732"

    viz = TopViewVisualizer(TARGET)
    if viz.load_data():
        viz.draw()
        viz.show()