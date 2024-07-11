import cv2
from ultralytics import YOLO
import json


class YOLOPose:
    def __init__(self, input_path, detections_file):
        self.input_path = input_path
        self.detections_file = detections_file
        self.detections = self.load_json(detections_file)

    def load_json(self, json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_json(self, data, json_path):
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def detect_poses(self):
        if self.detections:
            print("Using cached pose detections.")
            return

        model = YOLO('yolov8x-pose-p6.pt')
        cap = cv2.VideoCapture(self.input_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame_detections = []

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                keypoints = r.keypoints.data.cpu().numpy()  # Get keypoints

                for box, kps in zip(boxes, keypoints):
                    detection = {
                        "bbox": box.tolist(),  # [x1, y1, x2, y2]
                        "keypoints": kps.tolist()  # [[x1, y1, conf1], [x2, y2, conf2], ...]
                    }
                    frame_detections.append(detection)

            self.detections[frame_count] = frame_detections
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        self.save_json(self.detections, self.detections_file)
        print(f"Saved pose detections for {frame_count} frames.")