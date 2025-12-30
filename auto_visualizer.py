import matplotlib

# 强制使用 TkAgg 后端，修复 Qt 报错
try:
    matplotlib.use('TkAgg')
except:
    pass

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import os
from abc import ABC, abstractmethod

# --- 🎨 全局配色配置 ---
COLOR_PATH = '#444444'  # 路径连线颜色 (深灰)
COLOR_STAR = '#FF0000'  # 截图点 (红星)
CMAP_HEIGHT = 'viridis'  # 高度颜色映射 (蓝->绿->黄)


# ==========================================
# 0. 抽象基类 (通用逻辑)
# ==========================================
class MapVisualizer(ABC):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.csv_path = os.path.join(folder_path, "map_matrix.csv")
        self.snapshots_dir = os.path.join(folder_path, "map_snapshots")
        self.df = None
        self.fig = None

    def load_data(self):
        """通用数据加载"""
        if not os.path.exists(self.csv_path):
            print(f"❌ 找不到文件: {self.csv_path}")
            return False
        try:
            df = pd.read_csv(self.csv_path)
            self.df = df.dropna(subset=['X', 'Y', 'Z']).copy()
            # 转换数值类型
            for col in ['X', 'Y', 'Z']:
                self.df[col] = pd.to_numeric(self.df[col])
            # 标记截图点
            self.df['HasImage'] = self.df['Image_Filename'].notna() & (self.df['Image_Filename'].str.strip() != "")
            print(f"✅ 数据加载成功: {len(self.df)} 个点")

            # 全局 Z 轴范围 (供子类使用)
            self.z_min_global = self.df['Z'].min()
            self.z_max_global = self.df['Z'].max()
            return True
        except Exception as e:
            print(f"❌ 数据解析失败: {e}")
            return False

    def on_pick(self, event):
        """通用点击回调: 点击红星打开图片"""
        try:
            ind = event.ind[0]
            # 获取 artist 绑定的数据切片
            data_subset = event.artist.get_gid()
            if data_subset is not None:
                row = data_subset.iloc[ind]
                print(f"\n📍 [T:{row['Timestamp']}] ({row['X']:.0f}, {row['Y']:.0f}, {row['Z']:.0f})")

                img_name = row['Image_Filename']
                if pd.notna(img_name) and str(img_name).strip() != "":
                    img_path = os.path.join(self.snapshots_dir, str(img_name))
                    if os.path.exists(img_path):
                        print(f"   📸 打开截图: {img_name}")
                        os.startfile(img_path)
                    else:
                        print(f"   ⚠️ 丢失: {img_path}")
                else:
                    print("   (无截图)")
        except Exception as e:
            print(f"交互错误: {e}")

    @abstractmethod
    def draw(self):
        pass

    def show(self):
        plt.show()


# ==========================================
# MODE 1: 纯 2D 地形 (Top-Down)
# ==========================================
class Terrain2DVisualizer(MapVisualizer):
    def draw(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Mode 1: 2D Contour Map")
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        ax = self.fig.add_subplot(111)

        x, y, z = self.df['X'], self.df['Y'], self.df['Z']

        # 路径
        ax.plot(x, y, c=COLOR_PATH, alpha=0.3, lw=1, zorder=1)
        # 散点
        sc = ax.scatter(x, y, c=z, cmap=CMAP_HEIGHT, s=20, alpha=0.6, picker=5, zorder=2)
        sc.set_gid(self.df)
        # 红星
        df_img = self.df[self.df['HasImage']]
        if not df_img.empty:
            st = ax.scatter(df_img['X'], df_img['Y'], c=COLOR_STAR, marker='*', s=150, edgecolors='k', zorder=3,
                            picker=5)
            st.set_gid(df_img)

        plt.colorbar(sc, label='Height (Z)')
        ax.set_title("Mode 1: 2D Top View (Color = Height)")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)


# ==========================================
# MODE 2: 纯 3D 剖面 (Single 3D + Slider)
# (这就是你刚才说要保留的那个)
# ==========================================
class Layered3DVisualizer(MapVisualizer):
    def draw(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Mode 2: 3D Section View")
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.subplots_adjust(bottom=0.25)  # 留出底部给滑块

        self.ax = self.fig.add_subplot(111, projection='3d')

        # 滑块
        ax_min = self.fig.add_axes([0.2, 0.1, 0.65, 0.03])
        ax_max = self.fig.add_axes([0.2, 0.05, 0.65, 0.03])
        self.s_min = Slider(ax_min, 'Min Z', self.z_min_global, self.z_max_global, valinit=self.z_min_global)
        self.s_max = Slider(ax_max, 'Max Z', self.z_min_global, self.z_max_global, valinit=self.z_max_global)

        self.s_min.on_changed(self.update_plot)
        self.s_max.on_changed(self.update_plot)

        # 初始绘制
        self.update_plot(None)

    def update_plot(self, val):
        self.ax.clear()
        z_lower, z_upper = self.s_min.val, self.s_max.val
        if z_lower > z_upper: z_lower = z_upper

        # 过滤
        mask = (self.df['Z'] >= z_lower) & (self.df['Z'] <= z_upper)
        df_slice = self.df[mask]
        if df_slice.empty: return

        # 绘图 (这里用纯色或简单颜色，强调结构)
        self.ax.plot(df_slice['X'], df_slice['Y'], df_slice['Z'], c='blue', alpha=0.1, lw=0.5)
        # 使用灰色点，减少视觉干扰
        self.ax.scatter(df_slice['X'], df_slice['Y'], df_slice['Z'], c='#555555', s=15, alpha=0.4, picker=5)

        # 红星
        df_img = df_slice[df_slice['HasImage']]
        if not df_img.empty:
            st = self.ax.scatter(df_img['X'], df_img['Y'], df_img['Z'], c=COLOR_STAR, marker='*', s=120, edgecolors='k',
                                 picker=5, zorder=10)
            st.set_gid(df_img)

        self.ax.set_title(f"Mode 2: 3D Slicer - Z [{z_lower:.0f} ~ {z_upper:.0f}]")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        try:
            self.ax.set_box_aspect((1, 1, 0.5))
        except:
            pass
        self.fig.canvas.draw_idle()


# ==========================================
# MODE 3: 多视图工程分析 (Top + Side + 3D)
# (这是刚才那个很强的版本)
# ==========================================
class MultiViewVisualizer(MapVisualizer):
    def draw(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Mode 3: Engineering Multi-View")
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)

        # 3个子图
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Top
        self.ax2 = self.fig.add_subplot(2, 2, 3)  # Side (XZ)
        self.ax3 = self.fig.add_subplot(1, 2, 2, projection='3d')  # 3D

        # 滑块
        ax_min = self.fig.add_axes([0.15, 0.05, 0.3, 0.03])
        ax_max = self.fig.add_axes([0.55, 0.05, 0.3, 0.03])
        self.s_min = Slider(ax_min, 'Min Z', self.z_min_global, self.z_max_global, valinit=self.z_min_global)
        self.s_max = Slider(ax_max, 'Max Z', self.z_min_global, self.z_max_global, valinit=self.z_max_global)
        self.s_min.on_changed(self.update_plots)
        self.s_max.on_changed(self.update_plots)

        self.update_plots(None)

    def update_plots(self, val):
        z_lower, z_upper = self.s_min.val, self.s_max.val
        if z_lower > z_upper: z_lower = z_upper

        mask = (self.df['Z'] >= z_lower) & (self.df['Z'] <= z_upper)
        df_slice = self.df[mask]
        if df_slice.empty: return
        df_img = df_slice[df_slice['HasImage']]

        # 通用参数
        kwargs_sc = {'c': df_slice['Z'], 'cmap': CMAP_HEIGHT, 'vmin': self.z_min_global, 'vmax': self.z_max_global,
                     's': 15, 'alpha': 0.6, 'picker': 5}
        kwargs_st = {'c': COLOR_STAR, 'marker': '*', 's': 120, 'edgecolors': 'k', 'picker': 5, 'zorder': 10}

        # 1. Top View (XY)
        self.ax1.clear()
        self.ax1.plot(df_slice['X'], df_slice['Y'], c=COLOR_PATH, alpha=0.2)
        sc1 = self.ax1.scatter(df_slice['X'], df_slice['Y'], **kwargs_sc)
        sc1.set_gid(df_slice)
        if not df_img.empty:
            st1 = self.ax1.scatter(df_img['X'], df_img['Y'], **kwargs_st)
            st1.set_gid(df_img)
        self.ax1.set_title("1. Top View (XY)")
        self.ax1.set_ylabel('Y')
        self.ax1.grid(True, alpha=0.3)

        # 2. Side View (XZ) - 检查分层
        self.ax2.clear()
        self.ax2.scatter(df_slice['X'], df_slice['Z'], **kwargs_sc)
        self.ax2.set_title("2. Side View (XZ) - Check Layers")
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Z (Height)')
        self.ax2.grid(True, which='both', linestyle='--')

        # 3. 3D View
        self.ax3.clear()
        self.ax3.plot(df_slice['X'], df_slice['Y'], df_slice['Z'], c='blue', alpha=0.1)
        sc3 = self.ax3.scatter(df_slice['X'], df_slice['Y'], df_slice['Z'], **kwargs_sc)
        sc3.set_gid(df_slice)
        if not df_img.empty:
            st3 = self.ax3.scatter(df_img['X'], df_img['Y'], df_img['Z'], **kwargs_st)
            st3.set_gid(df_img)
        self.ax3.set_title("3. 3D View")
        try:
            self.ax3.set_box_aspect((1, 1, 0.4))
        except:
            pass

        self.fig.canvas.draw_idle()


# ==========================================
# MODE 4: 拓扑网络 (预留)
# ==========================================
class TopologyVisualizer(MapVisualizer):
    def draw(self):
        print("\n>>> Mode 4: Topology Graph")
        print("    此模式将忽略精确坐标，仅展示关键点(Node)的连接关系(Edge)。")
        print("    适用于传送门、非欧几里得空间分析。")
        print("    (目前为占位符，需引入 networkx 库实现)")

        self.fig = plt.figure(figsize=(8, 6))
        self.fig.canvas.manager.set_window_title("Mode 4: Topology (Placeholder)")
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Topology Graph\n(Coming Soon)", ha='center', va='center', fontsize=20)
        ax.axis('off')


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # ⚠️ 修改这里为你的路径
    TARGET_FOLDER = r"game_data_output/map_data_20251223_164356"

    # ⚠️ 切换模式 (1, 2, 3, 4)
    # Mode 1: 2D 地形 (适合看跑图路径)
    # Mode 2: 3D 剖面 (适合沉浸式看立体结构)
    # Mode 3: 多视图工程 (适合检查分层/空隙)
    # Mode 4: 拓扑 (预留)
    MODE = 3

    print(f"--- 启动可视化 (Mode {MODE}) ---")

    viz = None
    if MODE == 1:
        viz = Terrain2DVisualizer(TARGET_FOLDER)
    elif MODE == 2:
        viz = Layered3DVisualizer(TARGET_FOLDER)
    elif MODE == 3:
        viz = MultiViewVisualizer(TARGET_FOLDER)
    elif MODE == 4:
        viz = TopologyVisualizer(TARGET_FOLDER)

    if viz and viz.load_data():
        viz.draw()
        viz.show()