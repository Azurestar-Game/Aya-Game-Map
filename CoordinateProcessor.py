import re
import math


class CoordinateProcessor:
    def __init__(self, max_jump=5000, min_coord_val=1000):
        """
        :param max_jump: 允许的最大单帧移动距离（超过此值视为识别错误/瞬移）
        :param min_coord_val: 坐标最小绝对值（用于过滤类似 '49' 这种截断错误）
        """
        self.last_valid_pos = None
        self.max_jump = max_jump
        self.min_coord_val = min_coord_val

    def clean_data(self, raw_text):
        """
        核心清洗逻辑：处理 0X, 04, Z->2 等 OCR 杂质
        """
        # 1. 提取所有整数
        numbers = re.findall(r'-?\d+', raw_text)
        try:
            nums = [int(n) for n in numbers]
        except ValueError:
            return None

        # 2. 启发式修复逻辑

        # 修复 (a): 移除 'Z' 被误识别出的 '2'
        # 现象：数据变成4个，且第3个位置是2 (X, Y, 2, Z)
        # 或者数据杂乱，但包含单独的数字 '2' (且它显然不是坐标值)
        if len(nums) > 3:
            try:
                # 如果存在数字 2，且它不是最后一个数字（Z坐标），尝试移除它
                # 这里假设你的游戏坐标通常远大于 2
                idx_2 = nums.index(2)
                if idx_2 < len(nums) - 1:
                    nums.pop(idx_2)
            except ValueError:
                pass

        # 修复 (b): 移除开头的干扰项 (0X, 04 等)
        # 现象：第一个数字很小 (如 0, 4)，且移除它后长度刚好是 3
        if len(nums) > 3 and abs(nums[0]) < 100:
            nums.pop(0)

        # 3. 最终校验
        if len(nums) != 3:
            # 如果还不是3个，可能发生了数字拆分 (如 441: 96)，这种情况建议直接丢弃，
            # 因为很难自动判断是拼接还是真的有两个数。
            return None

        return nums

    def process(self, raw_text):
        """
        处理单行文本，返回: (状态字符串, 坐标列表)
        """
        current_pos = self.clean_data(raw_text)

        if current_pos is None:
            return "Format Error", None

        # 初始化检查：防止锁定到错误的初始值（如 '49'）
        if self.last_valid_pos is None:
            # 简单校验：坐标值不能太小 (防止锁定到 49 这种截断值)
            if abs(current_pos[0]) < self.min_coord_val:
                return "Init Ignore (Value too small)", None

            self.last_valid_pos = current_pos
            return "Init Success", current_pos

        # 距离过滤
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.last_valid_pos, current_pos)))

        if dist > self.max_jump:
            return f"Outlier (d={dist:.0f})", None  # 丢弃异常值
        else:
            self.last_valid_pos = current_pos  # 更新位置
            return "Valid", current_pos


# --- 测试演示 ---
processor = CoordinateProcessor(max_jump=8000)

# 选取你数据中的几个典型例子
test_lines = [
    "原始识别: 49,Y-21222,2:3891",  # 截断值，会被 Init Ignore
    "原始识别: 0X:44049,Y:-20679,2:3891",  # 0X 开头，Z->2
    "原始识别: 04:44196,Y:-19476,2:3891",  # 04 开头
    "原始识别: :49890,Y:-192942:9891",  # 数字粘连 (Outlier)
    "原始识别: 043924,Y:-18945,2:3891"  # 043924 这种连体字也能被正确提取
]

print(f"{'Raw Text':<35} | {'Result':<25} | {'Status'}")
print("-" * 80)
for line in test_lines:
    raw = line.split("原始识别:")[1].split("==>")[0].strip() if "原始识别" in line else line
    status, res = processor.process(raw)
    print(f"{raw[:32]:<35} | {str(res):<25} | {status}")