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
        self.save_interval = 10

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
            last_frame = max(map(int, self.detections.keys())) if self.detections else -1
            input_path = Path(self.input_path)

            if input_path.is_file():
                total_frames = int(cv2.VideoCapture(str(input_path)).get(cv2.CAP_PROP_FRAME_COUNT))
            elif input_path.is_dir():
                total_frames = len(list(input_path.glob('*.png')))
            else:
                raise ValueError("Input must be a video file or a directory containing PNG images.")

            if last_frame + 1 >= total_frames:
                print("All frames have been processed. Using cached pose detections.")
                return
            else:
                print(f"Continuing from frame {last_frame + 1}")
        else:
            last_frame = -1

        model = YOLO('yolov8x-pose-p6.pt')
        input_path = Path(self.input_path)

        if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            self.process_video(model, input_path, last_frame + 1)
        elif input_path.is_dir():
            self.process_image_directory(model, input_path, last_frame + 1)
        else:
            raise ValueError("Input must be a video file or a directory containing PNG images.")

        self.save_json(self.detections, self.detections_file)
        print(f"Saved pose detections for {len(self.detections)} frames/images.")

    def process_video(self, model, video_path, start_frame):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pb = tqdm.tqdm(total=total_frames, initial=start_frame, desc="Processing video frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.process_frame(model, frame, frame_count)
            frame_count += 1
            pb.update(1)

            if frame_count % self.save_interval == 0:
                self.save_json(self.detections, self.detections_file)

        cap.release()
        pb.close()

    def process_image_directory(self, model, dir_path, start_frame):
        image_files = sorted([f for f in dir_path.glob('*.png')])
        total_images = len(image_files)

        pbar = tqdm.tqdm(total=total_images, initial=start_frame, desc="Processing images")
        for idx, img_path in enumerate(image_files[start_frame:], start=start_frame):
            frame = cv2.imread(str(img_path))
            self.process_frame(model, frame, idx)
            pbar.update(1)

            if (idx + 1) % self.save_interval == 0:
                self.save_json(self.detections, self.detections_file)

        pbar.close()

    def process_frame(self, model, frame, frame_index):
        results = model(frame)
        frame_detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            keypoints = r.keypoints.data.cpu().numpy()

            for box, conf, kps in zip(boxes, confs, keypoints):
                detection = {
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "keypoints": kps.tolist()
                }
                frame_detections.append(detection)

        self.detections[str(frame_index)] = frame_detections
