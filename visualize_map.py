import matplotlib

# 1. 强制使用 TkAgg 后端，防止报错
try:
    matplotlib.use('TkAgg')
except:
    pass

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 配置 ---
PATH_COLOR = '#0000ff'  # 路径颜色
POINT_COLOR = '#00ff00'  # 点颜色


def visualize_map_interactive(folder_path):
    # 处理路径
    if not os.path.isabs(folder_path):
        folder_path = os.path.join(os.getcwd(), folder_path)

    csv_path = os.path.join(folder_path, "map_matrix.csv")
    snapshots_dir = os.path.join(folder_path, "map_snapshots")

    print(f"正在读取: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    # 读取数据
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 清洗数据
    df_clean = df.dropna(subset=['X', 'Y', 'Z']).copy()
    if df_clean.empty:
        print("⚠️ 数据为空或无效。")
        return

    print(f"✅ 加载 {len(df_clean)} 个坐标点。")

    # --- 核心修改：坐标映射 ---
    # 用户指定: X, Y 是平面，Z 是高度
    xs = df_clean['X']
    ys = df_clean['Y']
    zs = df_clean['Z']  # Height

    # 绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    fig.canvas.manager.set_window_title(f"Map Vis - {os.path.basename(folder_path)}")
    ax.set_title(f"Game Map (Z is Height)\nPoints: {len(df_clean)}")

    # 画路径线
    ax.plot(xs, ys, zs, c=PATH_COLOR, alpha=0.3, label='Path')

    # 画散点 (颜色根据 Z 轴/高度 变化)
    scatter = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=50, picker=5, edgecolor='k')

    # 设置轴标签
    ax.set_xlabel('Game X')
    ax.set_ylabel('Game Y')
    ax.set_zlabel('Game Z (Height)')

    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Height (Z)')

    # 交互点击事件
    def on_pick(event):
        ind = event.ind[0]
        row = df_clean.iloc[ind]

        timestamp = row['Timestamp']
        x, y, z = row['X'], row['Y'], row['Z']
        img_name = row['Image_Filename']

        print(f"\n📍 选中点:")
        print(f"   时间: {timestamp}")
        print(f"   坐标: X={x}, Y={y}, Z={z}")

        if pd.notna(img_name) and str(img_name).strip() != "":
            img_path = os.path.join(snapshots_dir, str(img_name))
            if os.path.exists(img_path):
                print(f"   📸 打开截图...")
                try:
                    os.startfile(img_path)
                except:
                    print("   ❌ 无法打开图片")
            else:
                print(f"   ⚠️ 图片不存在: {img_path}")
        else:
            print("   (无截图)")

    fig.canvas.mpl_connect('pick_event', on_pick)

    print("🚀 窗口已弹出！Z轴现在显示为高度。")
    plt.show()


if __name__ == "__main__":
    # ⚠️ 请确保这里的文件夹名字正确
    TARGET_FOLDER_NAME = r"game_data_output/map_data_20251223_143323"

    visualize_map_interactive(TARGET_FOLDER_NAME)