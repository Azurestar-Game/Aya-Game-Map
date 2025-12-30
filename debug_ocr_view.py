import time
import cv2
import mss
import numpy as np
import pytesseract
import win32gui
import ctypes

# --- 1. 配置区域 (保持和你主程序一致) ---
pytesseract.pytesseract.tesseract_cmd = r'D:\Interface\Tesseract\tesseract.exe'
GAME_WINDOW_TITLE = "G2_depart"

# OCR 区域偏移 (需要检查这里是否对齐)
OCR_OFFSET = {
    'top': 1057,
    'left': 20,
    'width': 180,
    'height': 17
}

# 颜色过滤阈值 (HSV) - 黄色文字
# 如果你的坐标是白色的，这里会导致画面全黑！
HSV_LOWER = np.array([20, 100, 100])
HSV_UPPER = np.array([40, 255, 255])

# --- 下面是调试逻辑 ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    ctypes.windll.user32.SetProcessDPIAware()


def get_window_rect(hwnd):
    try:
        rect = win32gui.GetWindowRect(hwnd)
        return {'left': rect[0], 'top': rect[1], 'width': rect[2] - rect[0], 'height': rect[3] - rect[1]}
    except:
        return None


def debug_loop():
    print("⏳ 正在寻找窗口...")
    hwnd = None
    while not hwnd:
        hwnd = win32gui.FindWindow(None, GAME_WINDOW_TITLE)
        # 如果标题不是完全匹配，尝试部分匹配
        if not hwnd:
            def callback(h, _):
                if win32gui.IsWindowVisible(h) and GAME_WINDOW_TITLE in win32gui.GetWindowText(h):
                    nonlocal hwnd
                    hwnd = h

            win32gui.EnumWindows(callback, None)
        time.sleep(1)

    print(f"✅ 锁定窗口: {hwnd}")
    print("🔴 正在打开调试视图... 按 'Q' 退出")

    with mss.mss() as sct:
        while True:
            rect = get_window_rect(hwnd)
            if not rect: continue

            # 计算 OCR 区域
            bbox = {
                'top': rect['top'] + OCR_OFFSET['top'],
                'left': rect['left'] + OCR_OFFSET['left'],
                'width': OCR_OFFSET['width'],
                'height': OCR_OFFSET['height']
            }

            try:
                # 1. 抓图
                img = np.array(sct.grab(bbox))

                # 2. 图像增强 (放大3倍)
                h, w = img.shape[:2]
                scale = 300
                upscaled = cv2.resize(img, (w * scale // 100, h * scale // 100), interpolation=cv2.INTER_CUBIC)

                # 3. 颜色过滤
                hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

                # 4. 识别
                text = pytesseract.image_to_string(mask,
                                                   config=r'--psm 7 -c tessedit_char_whitelist=XYZ:,-0123456789').strip()

                # --- 可视化输出 ---
                # 将 原图、二值化蒙版 拼在一起显示
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                combined = np.vstack([upscaled, mask_bgr])

                # 在图上写字显示识别结果
                cv2.putText(combined, f"Result: [{text}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("OCR Debug (Top: Raw, Bottom: Mask)", combined)
                print(f"识别结果: {text}")

            except Exception as e:
                print(f"❌ 报错: {e}")

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_loop()