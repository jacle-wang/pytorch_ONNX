import cv2
import numpy as np
from ultralytics import YOLO
import time
# 加载模型
model = YOLO('ONNX/Triangle_215_yolov8n_pretrain.onnx')
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 600)
IMAGE_PATH = '/mnt/d/code/wsl_coding/pytorch_ONNX/images/triangle_4.jpg'
VIDEO_PATH = '/mnt/d/code/wsl_coding/pytorch_ONNX/videos/triangle_7.mp4'
# 读取视频
cap = cv2.VideoCapture(VIDEO_PATH)
# 检测间隔：每 DETECT_EVERY 帧推理一次，其余帧沿用缓存结果
DETECT_EVERY = 10
frame_id = 0
# 缓存最近一次检测的框、置信度、类别、关键点
boxes_xyxy, confs, clss, kpts = [], [], [], []
names = model.names  # 类别索引 -> 类别名
start_time = time.time()
FPS = 0
# 逐帧读取并推理
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 只在检测帧推理，其余帧沿用缓存结果
    if frame_id % DETECT_EVERY == 0:
        end_time = time.time()
        # 计算 FPS,相隔时间内推理的帧数除以耗时（单位：帧/秒），FPS保留2位小数
        FPS =  round(DETECT_EVERY / (end_time - start_time), 2)
        start_time = time.time()
        r = model.predict(frame, task='pose', verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes_xyxy, confs, clss = [], [], []
        kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else []
    # 每帧都用缓存结果绘制，画面才连续（框每 DETECT_EVERY 帧更新一次）
    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, clss):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 画类别名 + 置信度
        cv2.putText(frame, f'{names[cls]} {conf:.2f} ', 
                    (int(x1), int(y1) - 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #在左上角画FPS
        cv2.putText(frame, f'FPS:{FPS}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for instance in kpts:
        for x, y, score in instance:
            if score > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow('img', frame)
    frame_id += 1
    # waitKey 必须放在循环内，按 q 退出 
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
 


 