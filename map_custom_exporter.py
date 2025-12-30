import matplotlib

try:
    matplotlib.use('Agg')  # 后台绘图
except:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os
import shutil

# ================= 配置区域 =================

# 1. 数据路径
TARGET_FOLDER = r"game_data_output/map_data_20251229_170732"

# 2. 自定义 Z 轴切片高度 (根据 Profile View 观察填写)
# 将生成 (-2000~0), (0~1500) ... 等区间的地图
CUSTOM_Z_LEVELS = [-20000, 20000]

# 3. 输出设置
OUTPUT_SIZE_PX = 2000
DPI = 100
OVERLAP = 50.0  # 稍微保留一点重叠，防止边界点断裂

# 4. 样式
COLORS_LIST = ['#D73027', '#FC8D59', '#00CED1', '#4575B4', '#08306B']
CMAP = LinearSegmentedColormap.from_list("Engineering", COLORS_LIST)
COLOR_SPECIAL = '#FF00FF'


# ===========================================

class CustomMapExporter:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.fixed_csv = os.path.join(folder_path, "map_matrix_fixed.csv")
        self.raw_csv = os.path.join(folder_path, "map_matrix.csv")
        self.output_dir = os.path.join(folder_path, "custom_maps_output")

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def load_data(self):
        target = self.fixed_csv if os.path.exists(self.fixed_csv) else self.raw_csv
        print(f"📂 加载数据: {target}")

        try:
            df = pd.read_csv(target)
            is_valid = df['X'].notna() & df['Y'].notna() & df['Z'].notna()
            df = df[is_valid].copy()
            for col in ['X', 'Y', 'Z']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 拓扑分段 (处理瞬移/暂停)
            df['segment_id'] = (~is_valid).astype(int).cumsum()

            if 'Image_Filename' in df.columns:
                df['HasImage'] = df['Image_Filename'].fillna("").astype(str).str.strip() != ""
            else:
                df['HasImage'] = False

            # 保存原始索引，用于后续判断连续性
            self.df = df
            self.indices = np.arange(len(df))
            self.np_x = df['X'].values
            self.np_y = df['Y'].values
            self.np_z = df['Z'].values
            self.np_seg = df['segment_id'].values
            self.np_has_img = df['HasImage'].values

            # 全局边界 (锁定 XY)
            pad = 200
            self.g_xmin, self.g_xmax = self.np_x.min() - pad, self.np_x.max() + pad
            self.g_ymin, self.g_ymax = self.np_y.min() - pad, self.np_y.max() + pad
            self.g_zmin, self.g_zmax = self.np_z.min(), self.np_z.max()

            print(f"✅ 数据加载完毕。点数: {len(df)}")
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

    def run_export(self, z_levels):
        levels = sorted(list(set(z_levels)))
        count = len(levels) - 1
        print(f"🚀 开始生成 {count} 张分层地图 (严格连续性模式)...")

        fig_size = OUTPUT_SIZE_PX / DPI

        for i in range(count):
            z_bottom = levels[i]
            z_top = levels[i + 1]

            # 1. 筛选本层数据
            # 这里的 mask 只是把点选出来了，但索引可能是不连续的
            mask = (self.np_z >= z_bottom - OVERLAP) & (self.np_z <= z_top + OVERLAP)

            if not np.any(mask):
                continue

            # 获取筛选后的数据子集
            plot_x = self.np_y[mask]  # East
            plot_y = self.np_x[mask]  # North
            plot_z = self.np_z[mask]

            # 【关键改进】获取这些点在原始数据中的索引
            sub_indices = self.indices[mask]
            sub_seg_ids = self.np_seg[mask]

            # 特殊点
            mask_st = self.np_has_img[mask]
            st_x = plot_x[mask_st]
            st_y = plot_y[mask_st]

            # 2. 创建画布
            fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=DPI)
            ax.set_xlim(self.g_ymin, self.g_ymax)
            ax.set_ylim(self.g_xmin, self.g_xmax)
            ax.set_aspect('equal')
            ax.axis('off')

            # 3. 绘制路径 (使用 LineCollection + 连续性检查)
            if len(plot_x) > 1:
                # 构建所有的点对线段: (P_i, P_i+1)
                points = np.array([plot_x, plot_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # 【核心逻辑优化】
                # 条件1: 原始索引必须连续 (index_diff == 1) -> 说明中间没有点跑到别的层去了
                # 条件2: 拓扑 ID 必须相同 -> 说明没有发生传送或重置

                idx_diffs = sub_indices[1:] - sub_indices[:-1]
                seg_diffs = sub_seg_ids[1:] == sub_seg_ids[:-1]

                valid_connections = (idx_diffs == 1) & seg_diffs

                # 只保留符合条件的线段
                clean_segments = segments[valid_connections]

                # 绘制
                if len(clean_segments) > 0:
                    lc = LineCollection(clean_segments, colors='#555555', linewidths=2.5, alpha=0.6)
                    ax.add_collection(lc)

            # 4. 绘制散点
            ax.scatter(plot_x, plot_y, c=plot_z, cmap=CMAP,
                       vmin=self.g_zmin, vmax=self.g_zmax,
                       s=50, alpha=0.9, edgecolors='none')

            # 5. 绘制特殊点
            if len(st_x) > 0:
                ax.scatter(st_x, st_y, c=COLOR_SPECIAL, marker='*', s=400,
                           edgecolors='black', linewidths=1.5, zorder=100)

            # 6. 水印
            label = "" # f"Layer {i + 1}: {z_bottom} to {z_top}"
            ax.text(0.02, 0.95, label, transform=ax.transAxes,
                    fontsize=24, color='black', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

            filename = f"Map_L{i + 1}_{int(z_bottom)}_{int(z_top)}.png"
            save_path = os.path.join(self.output_dir, filename)

            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"   -> 生成: {filename} (点数: {len(plot_x)} / 线段: {np.sum(valid_connections)})")

        print(f"\n✨ 严格连续性地图导出完成！: {self.output_dir}")


if __name__ == "__main__":
    exporter = CustomMapExporter(TARGET_FOLDER)
    if exporter.load_data():
        exporter.run_export(CUSTOM_Z_LEVELS)