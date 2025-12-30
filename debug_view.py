import time
import cv2
import mss
import numpy as np
import pytesseract
import re
import math

# --- 配置区域 ---
pytesseract.pytesseract.tesseract_cmd = r'D:\Interface\Tesseract\tesseract.exe'

# 新坐标: 793, 1220 -> 932, 1231
# mss 需要的是 {top, left, width, height}
COORDS_BBOX = {
    'top': 1220,
    'left': 793,
    'width': 932 - 793,  # = 139
    'height': 1231 - 1220 + 2  # = 11 + 2 (非常扁，下面代码里加了放大处理)
}


# --- 核心滤波类 (升级为基于时间的动态滤波) ---
class TimeBasedFilter:
    def __init__(self, max_speed=10000, base_tolerance=500):
        """
        :param max_speed: 游戏角色最大移动速度 (单位/秒)。
                          建议设大一点，比如正常跑是 2000，瞬移/冲刺算 10000。
        :param base_tolerance: 基础容差距离 (防止 dt 太小时阈值过小误杀 OCR 抖动)。
        """
        self.last_valid_pos = None
        self.last_valid_time = None

        self.max_speed = max_speed
        self.base_tolerance = base_tolerance
        self.min_coord_val = 1000  # 防止截断错误

    def clean_data(self, raw_text):
        """数据清洗逻辑 (保持不变)"""
        numbers = re.findall(r'-?\d+', raw_text)
        try:
            nums = [int(n) for n in numbers]
        except ValueError:
            return None

        # 修复杂质
        if len(nums) > 3:
            # 尝试移除单独的 '2' (Z轴误识别)
            try:
                if 2 in nums:
                    idx_2 = nums.index(2)
                    # 只有当它是中间的数字时才移除
                    if 0 < idx_2 < len(nums) - 1:
                        nums.pop(idx_2)
            except:
                pass

            # 尝试移除开头的干扰项 (如 0, 4)
            if len(nums) > 3 and abs(nums[0]) < 100:
                nums.pop(0)

        if len(nums) != 3:
            return None
        return nums

    def process(self, raw_text):
        """
        处理数据，引入时间变量
        """
        current_time = time.time()
        current_pos = self.clean_data(raw_text)

        if current_pos is None:
            return "❌ 格式错误", None

        # 初始化
        if self.last_valid_pos is None:
            if abs(current_pos[0]) < self.min_coord_val:
                return "⚠️ 初始值过小", None

            self.last_valid_pos = current_pos
            self.last_valid_time = current_time
            return "✅ 初始化", current_pos

        # --- 核心算法更新 ---
        # 1. 计算时间差 (dt)
        dt = current_time - self.last_valid_time
        if dt <= 0: dt = 0.001  # 防止除以0

        # 2. 计算实际移动距离
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.last_valid_pos, current_pos)))

        # 3. 计算动态阈值
        # 允许的最大距离 = (最大速度 * 时间差) + 基础容差
        # 例如：间隔 0.1秒，最大速度 10000 -> 允许跳变 1000 + 500 = 1500
        # 例如：卡顿 1.0秒，最大速度 10000 -> 允许跳变 10000 + 500 = 10500
        allowed_jump = (self.max_speed * dt) + self.base_tolerance

        # 4. 判定
        if dist > allowed_jump:
            # 计算一下当前异常数据的“瞬时速度”，方便调试
            curr_speed = dist / dt
            return f"🚫 速度异常 ({curr_speed:.0f}/s > {self.max_speed})", None
        else:
            self.last_valid_pos = current_pos
            self.last_valid_time = current_time
            return "✅ 正常", current_pos


# --- 主程序 ---
def debug_view():
    with mss.mss() as sct:
        print(">>> 调试模式 (显示完整窗口) <<<")
        print(f"抓取区域: {COORDS_BBOX}")
        print("按 'q' 键退出")
        print("-" * 50)

        # 初始化滤波器
        tracker = TimeBasedFilter(max_speed=15000, base_tolerance=1000)

        while True:
            # 1. 截图
            sct_img = sct.grab(COORDS_BBOX)
            img = np.array(sct_img)

            # --- 图像增强 (放大方便OCR) ---
            scale_percent = 300  # 放大 3 倍
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            img_upscaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

            # 2. 颜色提取
            hsv = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # 3. OCR 识别
            custom_config = r'--psm 7 -c tessedit_char_whitelist=XYZ:,-0123456789'
            text = pytesseract.image_to_string(mask, config=custom_config).strip()

            # 4. 滤波处理
            if text:
                status, coords = tracker.process(text)
                if coords:
                    # ⚠️ 修改点：去掉 [:15]，显示完整文本，这样你就能看到 Z 了
                    print(f"文本: {text:<25} | {status} | 坐标: {coords}")
                else:
                    print(f"文本: {text:<25} | {status}")

            # 5. 显示窗口 (显示两个，方便对比)
            # 原始截图（放大版）
            cv2.imshow('1. Raw Image (Upscaled)', img_upscaled)
            # OCR 看到的黑白图
            cv2.imshow('2. OCR Mask', mask)

            # 窗口位置调整 (可选，防止重叠)
            cv2.moveWindow('1. Raw Image (Upscaled)', 100, 100)
            cv2.moveWindow('2. OCR Mask', 600, 100)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_view()