import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import json
import os

class PersonSegmentation:
    def __init__(self, output_dir):
        self.yolo_model = YOLO('yolov8x-pose-p6.pt')
        self.sam_predictor = self.initialize_sam()
        self.output_dir = output_dir
        self.mask_dir = os.path.join(output_dir, 'masks')
        os.makedirs(self.mask_dir, exist_ok=True)
        self.detections = []
        self.detection_file = os.path.join(output_dir, 'detections.json')
        self.bg_video_path = os.path.join(output_dir, 'background_only.mp4')

    def initialize_sam(self):
        sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
        return SamPredictor(sam)

    def load_existing_data(self):
        if os.path.exists(self.detection_file):
            with open(self.detection_file, 'r') as f:
                self.detections = json.load(f)
            return True
        return False

    import numpy as np

    def process_frame(self, frame, frame_num):
        mask_file = os.path.join(self.mask_dir, f'mask_{frame_num:06d}.png')

        if frame_num < len(self.detections):
            frame_detections = self.detections[frame_num]
        else:
            # YOLO detection
            results = self.yolo_model(frame, classes=[0])  # 0 is the class index for person
            frame_detections = []
            for r in results:
                for box in r.boxes.data:
                    x1, y1, x2, y2, score, class_id = box.tolist()
                    if class_id == 0:  # Ensure it's a person
                        frame_detections.append([x1, y1, x2, y2])
            self.detections.append(frame_detections)

        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        else:
            # SAM segmentation
            self.sam_predictor.set_image(frame)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for box in frame_detections:
                box_array = np.array(box)  # Convert list to numpy array
                masks, _, _ = self.sam_predictor.predict(
                    box=box_array,
                    multimask_output=False
                )
                mask = np.logical_or(mask, masks[0]).astype(np.uint8)

            # Scale the mask to 255
            mask = mask * 255

            # Post-processing: remove small isolated regions and close gaps
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            cv2.imwrite(mask_file, mask)

        # Invert mask to get background
        bg_mask = cv2.bitwise_not(mask)

        # Apply mask to original frame
        bg_only = cv2.bitwise_and(frame, frame, mask=bg_mask)

        return bg_only, mask

    def process_video(self, input_path, force_reprocess=False):
        if not force_reprocess and self.load_existing_data():
            print("Loaded existing detection data.")
            if os.path.exists(self.bg_video_path):
                print(f"Background video already exists at {self.bg_video_path}")
                return True
            else:
                print("Existing detection data found, but background video is missing. Reprocessing video...")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open the input video at {input_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.bg_video_path, fourcc, fps, (width, height))

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bg_only, _ = self.process_frame(frame, frame_num)
            out.write(bg_only)

            frame_num += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Save detections to JSON file
        with open(self.detection_file, 'w') as f:
            json.dump(self.detections, f, indent=2)

        print(f"Processed {frame_num} frames")
        print(f"Masks saved in: {self.mask_dir}")
        print(f"Detections saved in: {self.detection_file}")
        print(f"Background video saved to: {self.bg_video_path}")
        return True

    def get_detections(self):
        return self.detections

    def get_bg_video_path(self):
        return self.bg_video_path