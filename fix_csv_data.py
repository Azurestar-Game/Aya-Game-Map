import matplotlib

# 强制使用 TkAgg 后端，修复 Qt 报错
try:
    matplotlib.use('TkAgg')
except:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置参数 ---
# ⚠️ 确保路径正确
ROOT = r"game_data_output/map_data_20251224_213933"
INPUT_FILE = os.path.join(ROOT, "map_matrix.csv")
OUTPUT_FILE = os.path.join(ROOT, "map_matrix_fixed.csv")

# --- 算法参数 ---
OUTLIER_THRESHOLD = 3000

# [核心修改] 加大窗口，增强抗噪稳定性
# 建议设为 15~21 (约 7-10秒的数据跨度)，这样短期的 OCR 抽风会被彻底无视
WINDOW_SIZE = 15


class TrajectoryRepairerV4:
    def __init__(self, df):
        self.df = df.sort_values('Timestamp').reset_index(drop=True)

    def generate_candidates(self, val):
        """生成 OCR 可能的错误变体"""
        if pd.isna(val): return []
        val_int = int(val)
        val_str = str(abs(val_int))
        candidates = set()

        candidates.add(val_int)  # 原值
        candidates.add(-val_int)  # 符号反转
        candidates.add(int(val_int / 10))  # 多读一位
        candidates.add(int(val_int * 10))  # 少读一位

        if len(val_str) > 2:
            try:
                sign = 1 if val_int >= 0 else -1
                candidates.add(sign * int(val_str[1:]))  # 去头
                candidates.add(int(val_int // 10))  # 去尾
            except:
                pass

        return list(candidates)

    def fix_axis(self, series_vals):
        """核心逻辑：基于宽窗口中位数 + 惯性锁定"""
        vals = series_vals.values
        n = len(vals)
        fixed_vals = np.copy(vals)

        # 1. 计算滑动中位数 (参考真值)
        series_pd = pd.Series(vals)
        median_guide = series_pd.rolling(window=WINDOW_SIZE, center=True, min_periods=1).median().values

        # 记录上一个有效值，用于惯性锁定
        last_valid_val = median_guide[0]

        for i in range(n):
            raw = vals[i]
            guide = median_guide[i]

            if pd.isna(raw):
                fixed_vals[i] = last_valid_val  # 简单填补
                continue

            # --- 阶段一：判断是否离群 ---
            diff_guide = abs(raw - guide)

            candidates = self.generate_candidates(raw)
            best_cand = raw

            # 如果原始值离参考线太远，说明肯定是错的，必须修
            if diff_guide > OUTLIER_THRESHOLD:
                min_dist_guide = float('inf')
                for cand in candidates:
                    d = abs(cand - guide)
                    if d < min_dist_guide:
                        min_dist_guide = d
                        best_cand = cand

                # 如果修完还是很离谱，强制归位到参考线
                if min_dist_guide > OUTLIER_THRESHOLD:
                    best_cand = guide

            # --- 阶段二：惯性锁定 (解决震荡问题) ---
            # 此时 best_cand 可能是修正后的值，也可能是原始值
            # 但有可能 candidates 里有两个值都离 guide 差不多远（或者 guide 本身在震荡）
            # 我们引入 "Last Value" 权重：优先保持不动

            # 如果 best_cand 和 last_valid_val 差距巨大 (比如跳层)，再次确认
            # 只有当 guide 也强烈支持跳变时，才允许跳变

            dist_to_last = abs(best_cand - last_valid_val)
            dist_guide_to_last = abs(guide - last_valid_val)

            # 逻辑：
            # 如果参考线(Guide)都很稳定(没跳)，但候选值(Best)想跳 -> 禁止跳，强制锁死在上一个值
            # 只有当参考线自己也跳了(说明是大势所趋)，才允许候选值跳
            if dist_guide_to_last < (OUTLIER_THRESHOLD / 2) and dist_to_last > OUTLIER_THRESHOLD:
                # 压制跳变
                final_val = last_valid_val
            else:
                final_val = best_cand

            fixed_vals[i] = final_val
            last_valid_val = final_val

        return fixed_vals

    def run(self):
        print(f"🚀 开始 V4 修复 (大窗口抗震版)...")
        fixed_data = {}

        for axis in ['X', 'Y', 'Z']:
            print(f"   处理 {axis} 轴...")
            # 线性插值填补空洞，为中位数计算提供支持
            raw_series = self.df[axis].interpolate(method='linear', limit_direction='both')
            fixed_data[axis] = self.fix_axis(raw_series)

        df_fixed = self.df.copy()
        df_fixed['X'] = fixed_data['X']
        df_fixed['Y'] = fixed_data['Y']
        df_fixed['Z'] = fixed_data['Z']

        return df_fixed


def plot_comparison(df_raw, df_fixed):
    plt.figure(figsize=(15, 10))
    limit = min(len(df_raw), 1000)  # 只画前1000个点看细节

    t_raw = df_raw['Timestamp'].iloc[:limit]
    t_raw = t_raw - t_raw.iloc[0]

    axes_list = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes_list):
        plt.subplot(3, 1, i + 1)
        # 原始数据
        plt.plot(t_raw, df_raw[axis].iloc[:limit], 'r.', markersize=3, label='Raw', alpha=0.3)
        # 修复数据
        plt.plot(t_raw, df_fixed[axis].iloc[:limit], 'b-', linewidth=1.5, label='Fixed V4')

        plt.title(f"{axis} Axis")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # 智能 Y 轴缩放
        valid = df_fixed[axis].iloc[:limit]
        if not valid.empty:
            plt.ylim(valid.min() - 2000, valid.max() + 2000)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        if os.path.exists("map_matrix.csv"):
            INPUT_FILE = "map_matrix.csv"
            OUTPUT_FILE = "map_matrix_fixed.csv"
        else:
            print(f"❌ 找不到文件: {INPUT_FILE}")
            exit()

    try:
        df_raw = pd.read_csv(INPUT_FILE)
        repairer = TrajectoryRepairerV4(df_raw)
        df_fixed = repairer.run()

        df_fixed.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ 修复完成！\n   输出: {OUTPUT_FILE}")

        plot_comparison(df_raw, df_fixed)

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()