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
        self.pose_mask_associations = {}
        self.pose_mask_file = os.path.join(output_dir, 'pose_mask_associations.json')
        self.person_reid = {}
        self.person_reid_file = os.path.join(output_dir, 'person_reid.json')

    def initialize_sam(self):
        sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
        return SamPredictor(sam)

    def load_existing_data(self):
        if os.path.exists(self.detection_file):
            with open(self.detection_file, 'r') as f:
                self.detections = json.load(f)
            return True
        return False

    def process_frame(self, frame, frame_num):
        composite_mask_file = os.path.join(self.mask_dir, f'mask_{frame_num:06d}.png')
        frame_detections = []
        frame_masks = []
        frame_colors = []

        if frame_num < len(self.detections):
            frame_detections = self.detections[frame_num]
        else:
            # YOLO detection
            results = self.yolo_model(frame, classes=[0])  # 0 is the class index for person
            for r in results:
                for box in r.boxes.data:
                    x1, y1, x2, y2, score, class_id = box.tolist()
                    if class_id == 0:  # Ensure it's a person
                        frame_detections.append([x1, y1, x2, y2])
            self.detections.append(frame_detections)

        # SAM segmentation
        self.sam_predictor.set_image(frame)
        composite_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for i, box in enumerate(frame_detections):
            box_array = np.array(box)
            masks, _, _ = self.sam_predictor.predict(
                box=box_array,
                multimask_output=False
            )
            mask = masks[0].astype(np.uint8)

            # Save individual mask
            individual_mask_file = os.path.join(self.mask_dir, f'mask_{frame_num:06d}-{i}.png')
            cv2.imwrite(individual_mask_file, mask * 255)

            # Update composite mask
            composite_mask = np.logical_or(composite_mask, mask).astype(np.uint8)

            # Calculate average color
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            avg_color = np.mean(masked_frame[mask > 0], axis=0).tolist()

            frame_masks.append(individual_mask_file)
            frame_colors.append(avg_color)

        # Save composite mask
        cv2.imwrite(composite_mask_file, composite_mask * 255)

        # Update pose-mask associations
        self.pose_mask_associations[frame_num] = {
            'detections': frame_detections,
            'masks': frame_masks,
            'colors': frame_colors
        }

        # Invert mask to get background
        bg_mask = cv2.bitwise_not(composite_mask * 255)

        # Apply mask to original frame
        bg_only = cv2.bitwise_and(frame, frame, mask=bg_mask)

        return bg_only, composite_mask

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

        # Perform ReID
        self.perform_reid()

        # Save all data
        self.save_data()

        print(f"Processed {frame_num} frames")
        print(f"Masks saved in: {self.mask_dir}")
        print(f"Detections saved in: {self.detection_file}")
        print(f"Pose-mask associations saved in: {self.pose_mask_file}")
        print(f"Person ReID data saved in: {self.person_reid_file}")
        print(f"Background video saved to: {self.bg_video_path}")
        return True

    def perform_reid(self):
        all_colors = []
        all_positions = []
        all_sizes = []
        frame_indices = []
        detection_indices = []

        for frame_num, frame_data in self.pose_mask_associations.items():
            for i, (detection, color) in enumerate(zip(frame_data['detections'], frame_data['colors'])):
                all_colors.append(color)
                center_x = (detection[0] + detection[2]) / 2
                center_y = (detection[1] + detection[3]) / 2
                all_positions.append([center_x, center_y])
                size = (detection[2] - detection[0]) * (detection[3] - detection[1])
                all_sizes.append(size)
                frame_indices.append(frame_num)
                detection_indices.append(i)

        all_colors = np.array(all_colors)
        all_positions = np.array(all_positions)
        all_sizes = np.array(all_sizes)

        # Normalize features
        all_colors /= 255.0
        all_positions /= np.max(all_positions)
        all_sizes /= np.max(all_sizes)

        # Combine features
        features = np.hstack([all_colors, all_positions, all_sizes.reshape(-1, 1)])

        # Perform clustering
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.1, min_samples=2).fit(features)

        labels = clustering.labels_

        # Assign person IDs
        for label, frame_num, detection_index in zip(labels, frame_indices, detection_indices):
            if label == -1:
                continue  # Skip noise points
            person_id = f"person_{label}"
            if person_id not in self.person_reid:
                self.person_reid[person_id] = {}
            if frame_num not in self.person_reid[person_id]:
                self.person_reid[person_id][frame_num] = []
            self.person_reid[person_id][frame_num].append(detection_index)

    def save_data(self):
        with open(self.detection_file, 'w') as f:
            json.dump(self.detections, f, indent=2)

        with open(self.pose_mask_file, 'w') as f:
            json.dump(self.pose_mask_associations, f, indent=2)

        with open(self.person_reid_file, 'w') as f:
            json.dump(self.person_reid, f, indent=2)

    def get_detections(self):
        return self.detections

    def get_bg_video_path(self):
        return self.bg_video_path