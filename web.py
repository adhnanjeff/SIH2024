from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
import time
from sort import *
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  

model = YOLO("../Yolo-Weights/yolov8l.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cap = cv2.VideoCapture("Video.mp4")
mask = cv2.imread("mask.png")
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []
Count = 0
start = [450, 347, 620, 347]  # Start line coordinates
end = [300, 447, 600, 447]  # End line coordinates
prev_frame_time = 0

def generate_frames():
    global prev_frame_time, Count
    while True:
        success, img = cap.read()
        if not success:
            break

        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        imgRegion = cv2.bitwise_and(img, mask_resized)

        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

        results = model(imgRegion, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                    w, h = x2 - x1, y2 - y1
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        cv2.line(img, (start[0], start[1]), (start[2], start[3]), (255, 0, 0), 5)  # Start line
        cv2.line(img, (end[0], end[1]), (end[2], end[3]), (255, 0, 0), 5)  # End line

        for result in resultsTracker:
            x1, y1, x2, y2, Id = map(int, result)
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2  # Center of the bounding box

            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(0, 0, 255))
            cvzone.putTextRect(img, f' {int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

            if start[0] < cx < start[2] and start[1] - 15 < cy < start[1] + 15:
                if totalCount.count(Id) == 0:
                    totalCount.append(Id)
                    Count += 1
                    cv2.line(img, (start[0], start[1]), (start[2], start[3]), (0, 255, 0), 5)

            if end[0] < cx < end[2] and end[1] - 15 < cy < end[1] + 15:
                if totalCount.count(Id) != 0:
                    totalCount.remove(Id)
                    Count -= 1
                    cv2.line(img, (end[0], end[1]), (end[2], end[3]), (255, 0, 0), 5)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        #cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(img, f'     {Count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 255), 4)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
