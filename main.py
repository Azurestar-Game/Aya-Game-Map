import time
import cv2
import mss
import numpy as np
import pytesseract
import keyboard
import networkx as nx
import re
import os
import matplotlib.pyplot as plt

# --- 配置区域 ---
# Tesseract 安装路径 (Windows下通常需要指定)
pytesseract.pytesseract.tesseract_cmd = r'D:\Interface\Tesseract\tesseract.exe'

# 坐标区域 (根据你的截图估算的，实际需要你微调 bbox)
# left, top, width, height. 你可以用画图工具量一下像素
COORDS_BBOX = {'top': 980, 'left': 30, 'width': 400, 'height': 50}

# 阈值设置
MOVE_THRESHOLD = 50.0  # 移动超过多少距离记录一个新节点 (单位：游戏坐标单位)
MERGE_THRESHOLD = 20.0  # 靠近已有节点多少距离视为“同一个点”


class GameMapper:
    def __init__(self):
        self.sct = mss.mss()
        self.graph = nx.Graph()
        self.pos_history = []  # 记录轨迹 [(x,y,z), ...]
        self.node_counter = 0
        self.current_pos = None
        self.last_recorded_pos = None

        # 创建截图保存目录
        if not os.path.exists("./map_snapshots"):
            os.makedirs("./map_snapshots")

        print("Mapper Started. Press 'F7' to snapshot. Press 'Q' to quit.")

    def process_image(self, img):
        """图像预处理：提取黄色文字"""
        # 转换为 HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 定义黄色的范围 (根据截图大概估算，不准可以用吸管工具取色)
        # OpenCV中 H: 0-179, S: 0-255, V: 0-255
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # 膨胀一下让字更粗，方便OCR
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def get_coordinates(self):
        """截屏并识别坐标"""
        # 截取坐标区域
        sct_img = self.sct.grab(COORDS_BBOX)
        img = np.array(sct_img)

        # 预处理
        processed_img = self.process_image(img)  # 变成黑底白字

        # OCR 识别
        # psm 7 表示把图像视为单行文本
        text = pytesseract.image_to_string(processed_img, config='--psm 7 digits')

        # 解析文本 [X:43235,Y:-19794,Z:3891]
        # 正则表达式提取数字 (支持负号)
        matches = re.findall(r'(-?\d+)', text)

        if len(matches) >= 3:
            try:
                x, y, z = map(int, matches[:3])
                return np.array([x, y, z])
            except ValueError:
                return None
        return None

    def add_node(self, pos, is_keyframe=False):
        """添加节点到图结构"""
        node_id = self.node_counter

        # 检查是否可以合并到现有节点 (简单的最近邻搜索)
        # 实际大量数据可以用 KD-Tree 优化，这里暴力循环即可
        merged = False
        for existing_id, data in self.graph.nodes(data=True):
            existing_pos = np.array(data['pos'])
            dist = np.linalg.norm(pos - existing_pos)
            if dist < MERGE_THRESHOLD:
                node_id = existing_id
                merged = True
                break

        if not merged:
            self.graph.add_node(node_id, pos=pos, type='auto' if not is_keyframe else 'key')
            self.node_counter += 1
            print(f"New Node: {node_id} at {pos}")
        else:
            # 如果是关键帧，更新该节点属性
            if is_keyframe:
                self.graph.nodes[node_id]['type'] = 'key'
                print(f"Merged into Node: {node_id}")

        # 如果有上一个点，连接边
        if self.last_recorded_pos is not None:
            # 找到上一个点对应的ID (这里简化处理，实际需要记录last_node_id)
            # 为了演示，我们假设总是连接到最新的路径
            pass  # 实际代码需要维护 last_node_id

        return node_id

    def save_snapshot(self, node_id):
        """F7按下时，保存缩略图"""
        # 全屏截图
        full_screen = self.sct.grab(self.sct.monitors[1])
        img = np.array(full_screen)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 压缩缩略图 (比如宽 320)
        h, w = img.shape[:2]
        scale = 320 / w
        dim = (320, int(h * scale))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        filename = f"./map_snapshots/node_{node_id}.jpg"
        cv2.imwrite(filename, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        print(f"Snapshot saved: {filename}")

        # 在图中记录图片路径
        self.graph.nodes[node_id]['image'] = filename

    def run(self):
        last_check_time = 0
        self.last_node_id = -1

        while True:
            if keyboard.is_pressed('q'):
                break

            # 限制OCR频率，比如每0.5秒一次，太快没必要且占CPU
            if time.time() - last_check_time > 0.5:
                pos = self.get_coordinates()
                if pos is not None:
                    self.current_pos = pos

                    # 逻辑1：自动铺路
                    dist = 0
                    if self.last_recorded_pos is not None:
                        dist = np.linalg.norm(pos - self.last_recorded_pos)

                    if self.last_recorded_pos is None or dist > MOVE_THRESHOLD:
                        # 只有移动距离够大才记录新点
                        new_id = self.add_node(pos)

                        # 连接边
                        if self.last_node_id != -1:
                            self.graph.add_edge(self.last_node_id, new_id, weight=dist)

                        self.last_recorded_pos = pos
                        self.last_node_id = new_id

                last_check_time = time.time()

            # 逻辑2：按键截图
            if keyboard.is_pressed('f7'):
                if self.current_pos is not None:
                    print("F7 Pressed! Capturing keyframe...")
                    # 强制在当前位置记录一个点（即使距离不够）
                    key_node_id = self.add_node(self.current_pos, is_keyframe=True)
                    # 连线
                    if self.last_node_id != -1 and self.last_node_id != key_node_id:
                        self.graph.add_edge(self.last_node_id, key_node_id)

                    self.save_snapshot(key_node_id)
                    self.last_node_id = key_node_id  # 更新当前驻点
                    self.last_recorded_pos = self.current_pos

                    time.sleep(0.5)  # 防止按键连击

        # 结束时绘制
        self.plot_map()

    def plot_map(self):
        print("Plotting map...")
        # 提取位置用于绘图 (只画 X, Z 平面，忽略高度 Y，或者根据游戏轴向调整)
        # 假设 X是横向, Z是纵向
        pos_dict = {n: (d['pos'][0], d['pos'][2]) for n, d in self.graph.nodes(data=True)}

        colors = ['red' if d.get('type') == 'key' else 'blue' for n, d in self.graph.nodes(data=True)]

        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, pos_dict, node_size=20, node_color=colors, with_labels=False, alpha=0.6)
        plt.title("Game Topology Map")
        plt.show()


if __name__ == "__main__":
    mapper = GameMapper()
    mapper.run()