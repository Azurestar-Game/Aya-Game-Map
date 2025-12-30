import cv2
import mss
import numpy as np
import win32gui
import win32api
import ctypes

# --- 1. 防止 DPI 缩放导致画面模糊或错位 ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    ctypes.windll.user32.SetProcessDPIAware()

# ⚠️ 这里填你的游戏标题 (部分即可)
TARGET_TITLE = "G2_depart"

# --- 2. 初始状态 ---
CURRENT_RECT = {
    'left': 50,  # X
    'top': 50,  # Y
    'width': 150,  # W
    'height': 20  # H
}


def get_window_rect(hwnd):
    try:
        rect = win32gui.GetWindowRect(hwnd)
        return {
            'left': rect[0], 'top': rect[1],
            'width': rect[2] - rect[0], 'height': rect[3] - rect[1]
        }
    except:
        return None


def calibration_tool():
    print(f"⏳ 正在搜索窗口: [{TARGET_TITLE}]...")
    hwnd = None
    while not hwnd:
        def callback(h, _):
            if win32gui.IsWindowVisible(h) and TARGET_TITLE in win32gui.GetWindowText(h):
                nonlocal hwnd
                hwnd = h

        win32gui.EnumWindows(callback, None)
        if not hwnd: cv2.waitKey(1000)

    print(f"✅ 已锁定窗口句柄: {hwnd}")
    print("\n" + "=" * 50)
    print("【🎮 V2.3 自由窗口版】")
    print("  鼠标拖动      : ✅ 现在可以自由移动窗口了")
    print("  WASD / 方向键 : 移动绿框")
    print("  TFGH          : 调整尺寸 (T/G=高矮, F/H=宽窄)")
    print("  Shift (按住)  : 🚀 加速")
    print("  Enter         : ✅ 确认并输出结果")
    print("  Q             : 退出")
    print("=" * 50)

    # --- 窗口初始化 (只设置一次位置，不再锁定) ---
    window_name = '1. Global View (Green Box)'
    zoom_window_name = '2. Pixel Zoom (400%)'

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    cv2.namedWindow(zoom_window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(zoom_window_name, cv2.WND_PROP_TOPMOST, 1)

    # 初始位置设定 (运行后你可以随意拖走)
    cv2.moveWindow(window_name, 100, 100)
    cv2.moveWindow(zoom_window_name, 600, 100)

    with mss.mss() as sct:
        while True:
            # 获取窗口位置
            win_rect = get_window_rect(hwnd)
            if not win_rect: continue

            # 截取游戏画面
            safe_bbox = {
                'top': win_rect['top'],
                'left': win_rect['left'],
                'width': win_rect['width'],
                'height': win_rect['height']
            }

            if safe_bbox['left'] < 0: safe_bbox['left'] = 0
            if safe_bbox['top'] < 0: safe_bbox['top'] = 0

            try:
                img = np.array(sct.grab(safe_bbox))
            except:
                continue

            # --- 坐标限制 ---
            h_img, w_img = img.shape[:2]
            CURRENT_RECT['left'] = max(0, min(CURRENT_RECT['left'], w_img - CURRENT_RECT['width']))
            CURRENT_RECT['top'] = max(0, min(CURRENT_RECT['top'], h_img - CURRENT_RECT['height']))

            x, y = CURRENT_RECT['left'], CURRENT_RECT['top']
            w, h = CURRENT_RECT['width'], CURRENT_RECT['height']

            # --- 生成放大镜视图 (Zoom View) ---
            crop_img = img[y:y + h, x:x + w]
            if crop_img.size > 0:
                zoom_scale = 4
                zoom_img = cv2.resize(crop_img, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(zoom_img, (0, 0), (zoom_img.shape[1] - 1, zoom_img.shape[0] - 1), (0, 255, 255), 2)
                cv2.imshow(zoom_window_name, zoom_img)

            # --- 绘制主视图 ---
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(img, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 0), 1)
            cv2.line(img, (x + w // 2, y), (x + w // 2, y + h), (0, 255, 0), 1)

            info_text = f"X:{x} Y:{y} W:{w} H:{h}"
            cv2.rectangle(img, (x, y - 25), (x + 200, y), (0, 0, 255), -1)
            cv2.putText(img, info_text, (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, img)

            # --- 按键控制 ---
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): break

            is_shift = win32api.GetAsyncKeyState(0x10) & 0x8000
            speed = 10 if is_shift else 1

            if key == ord('w'): CURRENT_RECT['top'] -= speed
            if key == ord('s'): CURRENT_RECT['top'] += speed
            if key == ord('a'): CURRENT_RECT['left'] -= speed
            if key == ord('d'): CURRENT_RECT['left'] += speed

            if key == ord('t'): CURRENT_RECT['height'] -= speed
            if key == ord('g'): CURRENT_RECT['height'] += speed
            if key == ord('f'): CURRENT_RECT['width'] -= speed
            if key == ord('h'): CURRENT_RECT['width'] += speed

            CURRENT_RECT['width'] = max(10, CURRENT_RECT['width'])
            CURRENT_RECT['height'] = max(5, CURRENT_RECT['height'])

            if key == 13:  # Enter
                print("\n" + "✅" * 20)
                print("【校准成功】请复制下面的代码替换 OCR_OFFSET:")
                print(f"OCR_OFFSET = {{")
                print(f"    'top': {CURRENT_RECT['top']},")
                print(f"    'left': {CURRENT_RECT['left']},")
                print(f"    'width': {CURRENT_RECT['width']},")
                print(f"    'height': {CURRENT_RECT['height']}")
                print(f"}}")
                print("✅" * 20 + "\n")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibration_tool()