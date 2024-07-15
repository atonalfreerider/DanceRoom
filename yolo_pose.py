import cv2
from ultralytics import YOLO
import json
from pathlib import Path
import tqdm


class YOLOPose:
    def __init__(self, input_path, detections_file):
        self.input_path = input_path
        self.detections_file = detections_file
        self.detections = self.load_json(detections_file)

    @staticmethod
    def load_json(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    @staticmethod
    def save_json(data, json_path):
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def detect_poses(self):
        if self.detections:
            print("Using cached pose detections.")
            return

        model = YOLO('yolov8x-pose-p6.pt')
        input_path = Path(self.input_path)

        if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            self.process_video(model, input_path)
        elif input_path.is_dir():
            self.process_image_directory(model, input_path)
        else:
            raise ValueError("Input must be a video file or a directory containing PNG images.")

        self.save_json(self.detections, self.detections_file)
        print(f"Saved pose detections for {len(self.detections)} frames/images.")

    def process_video(self, model, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0

        pb = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.process_frame(model, frame, frame_count)
            frame_count += 1
            pb.update(1)

        cap.release()
        pb.close()

    def process_image_directory(self, model, dir_path):
        image_files = sorted([f for f in dir_path.glob('*.png')])

        pbar = tqdm.tqdm(total=len(image_files), desc="Processing images")
        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            self.process_frame(model, frame, idx)
            pbar.update(1)

        pbar.close()

    def process_frame(self, model, frame, frame_index):
        results = model(frame)
        frame_detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            keypoints = r.keypoints.data.cpu().numpy()

            for box, kps in zip(boxes, keypoints):
                detection = {
                    "bbox": box.tolist(),
                    "keypoints": kps.tolist()
                }
                frame_detections.append(detection)

        self.detections[str(frame_index)] = frame_detections
