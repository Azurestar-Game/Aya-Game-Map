import matplotlib

# 使用 Agg 后端，不需要显示窗口，专门用于后台生成图片
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os

# ==========================================
# 🛠️ 配置区域
# ==========================================
# 输出分辨率 (5000x5000 px)
IMG_SIZE_INCH = 50
DPI = 100

# 颜色配置 (保持一致)
COLORS_LIST = ['#D73027', '#FC8D59', '#fee090', '#e0f3f8', '#91bfdb', '#4575B4']
CMAP_HEIGHT = LinearSegmentedColormap.from_list("Engineering", COLORS_LIST)
COLOR_PATH = '#AAAAAA'  # 路径线条颜色 (淡灰色)
COLOR_SPECIAL = '#FF00FF'  # 图片点颜色


# ==========================================
# 1. 核心分层逻辑 (复用之前的逻辑)
# ==========================================
def get_layer_name(x, y, z, source_folder):
    if z < -10800: return "Monster"
    if z < -9200: return "Level 5"
    if z < -6000: return "Level 4"
    if z < -4200: return "Level 3"
    if z < -1500:
        if z > -2500:
            if y < 6500:
                return "Level 2"
            else:
                return "Level 1"
        else:
            return "Level 2"
    if z > -200: return "Entrance"
    return "Level 1"


# ==========================================
# 2. 导出器类
# ==========================================
class MapExporter:
    def __init__(self, target_list, output_dir="output_layers"):
        self.target_list = target_list
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_data(self):
        all_dfs = []
        global_seg_offset = 0
        print("📦 [Step 1/3] Loading Data...")

        for folder_path in self.target_list:
            if not os.path.exists(folder_path): continue
            fixed = os.path.join(folder_path, "map_matrix_fixed.csv")
            raw = os.path.join(folder_path, "map_matrix.csv")
            fpath = fixed if os.path.exists(fixed) else raw
            if not os.path.exists(fpath): continue

            try:
                sub_df = pd.read_csv(fpath)
                sub_df = sub_df[sub_df['X'].notna() & sub_df['Y'].notna() & sub_df['Z'].notna()].copy()
                if 'segment_id' not in sub_df.columns: sub_df['segment_id'] = 0
                sub_df['segment_id'] += global_seg_offset

                # 标记是否有图 (用于画星星)
                if 'Image_Filename' in sub_df.columns:
                    sub_df['HasImage'] = sub_df['Image_Filename'].fillna("").str.strip() != ""
                else:
                    sub_df['HasImage'] = False

                sub_df['Source'] = os.path.basename(folder_path)
                sub_df['Layer'] = sub_df.apply(lambda r: get_layer_name(r['X'], r['Y'], r['Z'], r['Source']), axis=1)

                if not sub_df.empty: global_seg_offset = sub_df['segment_id'].max() + 100
                all_dfs.append(sub_df)
                print(f"   ✅ Loaded: {sub_df['Source'].iloc[0]}")
            except:
                pass

        if not all_dfs: return False
        self.df = pd.concat(all_dfs, ignore_index=True)

        # 准备 Numpy 数据
        self.np_x = pd.to_numeric(self.df['X']).values
        self.np_y = pd.to_numeric(self.df['Y']).values
        self.np_z = pd.to_numeric(self.df['Z']).values
        self.np_seg = self.df['segment_id'].values
        self.np_layer = self.df['Layer'].values
        self.np_has_img = self.df['HasImage'].values
        self.connect_mask = (self.np_seg[:-1] == self.np_seg[1:])

        # 计算并锁定全局边界 (关键步骤)
        # 稍微留一点边距 (Padding)
        pad = 200
        self.g_xmin, self.g_xmax = self.np_x.min() - pad, self.np_x.max() + pad
        self.g_ymin, self.g_ymax = self.np_y.min() - pad, self.np_y.max() + pad

        # Z轴范围用于统一颜色映射
        self.g_zmin, self.g_zmax = self.np_z.min(), self.np_z.max()

        print(
            f"   📐 Global Bounds Locked: X[{self.g_xmin:.1f}, {self.g_xmax:.1f}], Y[{self.g_ymin:.1f}, {self.g_ymax:.1f}]")
        return True

    def export_all_layers(self):
        layers = ["Entrance", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Monster"]

        print("\n🎨 [Step 2/3] Rendering Layers (5000x5000px)...")

        # 还要生成一张包含所有的总图
        self.render_layer("All_Combined", None)

        for layer_name in layers:
            self.render_layer(layer_name, layer_name)

        print(f"\n✅ [Step 3/3] Done! Images saved in '{self.output_dir}'")

    def render_layer(self, file_prefix, target_layer):
        print(f"   ... Rendering: {file_prefix}")

        # 1. 创建超大画布
        fig = plt.figure(figsize=(IMG_SIZE_INCH, IMG_SIZE_INCH), dpi=DPI)
        # 创建一个占满整个图的 Axes，去掉边距
        ax = fig.add_axes([0, 0, 1, 1])

        # 2. 筛选数据
        if target_layer:
            mask = (self.np_layer == target_layer)
        else:
            mask = np.ones_like(self.np_layer, dtype=bool)  # All

        mask_lines = mask[:-1] & mask[1:] & self.connect_mask

        vis_x = self.np_x[mask]
        vis_y = self.np_y[mask]
        vis_z = self.np_z[mask]

        # 3. 绘制内容
        # 如果该层没数据，也得画一个空图，保证文件存在且尺寸对齐
        if len(vis_x) > 0:
            # 3.1 绘制线条 (底色)
            # 注意: Matplotlib plot x=East(Y), y=North(X)
            segs = np.stack([np.column_stack([self.np_y[:-1][mask_lines], self.np_y[1:][mask_lines]]),
                             np.column_stack([self.np_x[:-1][mask_lines], self.np_x[1:][mask_lines]])], axis=2)
            lc = LineCollection(segs, linewidths=2.0, colors=COLOR_PATH, alpha=0.6)
            ax.add_collection(lc)

            # 3.2 绘制点 (按高度着色)
            sc = ax.scatter(vis_y, vis_x, c=vis_z, cmap=CMAP_HEIGHT, s=80, alpha=0.9, edgecolors='none')
            sc.set_clim(self.g_zmin, self.g_zmax)  # 锁定颜色映射范围，保证所有图层颜色统一

            # 3.3 绘制截图标记 (星星)
            mask_st = mask & self.np_has_img
            if np.any(mask_st):
                st_x = self.np_x[mask_st]
                st_y = self.np_y[mask_st]
                ax.scatter(st_y, st_x, marker='*', s=600, c=COLOR_SPECIAL, edgecolors='black', linewidths=1.5,
                           zorder=100)

        # 4. 关键：锁定坐标轴以对其
        ax.set_xlim(self.g_ymin, self.g_ymax)  # Y data -> X axis
        ax.set_ylim(self.g_xmin, self.g_xmax)  # X data -> Y axis
        ax.set_aspect('equal')

        # 5. 移除所有装饰 (透明底必备)
        ax.axis('off')

        # 6. 保存
        save_path = os.path.join(self.output_dir, f"Layer_{file_prefix}.png")
        plt.savefig(save_path, transparent=True, dpi=DPI)
        plt.close(fig)  # 释放内存


# ==========================================
# 3. 执行入口
# ==========================================
if __name__ == "__main__":
    # 配置你的数据源路径
    TARGETS = [
        r"game_data_output/map_data_20251224_140944",
        r"game_data_output/map_data_20251224_152637",
        r"game_data_output/map_data_20251224_163308",
        r"game_data_output/map_data_20251224_185559",
        r"game_data_output/map_data_20251224_200336",
        r"game_data_output/map_data_20251224_211137",
        r"game_data_output/map_data_20251224_213933",
    ]

    exporter = MapExporter(TARGETS)
    if exporter.load_data():
        exporter.export_all_layers()