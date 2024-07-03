import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import json
import os
from sklearn.cluster import DBSCAN


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
        self.reid_video_path = os.path.join(output_dir, 'reid_video.mp4')
        self.pose_mask_associations = {}
        self.pose_mask_file = os.path.join(output_dir, 'pose_mask_associations.json')
        self.person_reid = {}
        self.person_reid_file = os.path.join(output_dir, 'person_reid.json')

    @staticmethod
    def initialize_sam():
        sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
        return SamPredictor(sam)

    def load_existing_data(self):
        if os.path.exists(self.detection_file):
            with open(self.detection_file, 'r') as f:
                self.detections = json.load(f)
            return True
        return False

    @staticmethod
    def get_body_part_colors(frame, mask, keypoints):
        # Define body parts
        chest_back_indices = [5, 6, 11, 12]  # shoulders and hips
        legs_indices = [13, 14, 15, 16]  # knees and ankles
        arms_indices = [7, 8, 9, 10]  # elbows and wrists

        def get_average_color(points):
            colors = []
            for x, y, conf in points:
                if conf > 0.5:  # Only consider high confidence keypoints
                    x, y = int(x), int(y)
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        roi = frame[max(0, y - 10):min(frame.shape[0], y + 10),
                              max(0, x - 10):min(frame.shape[1], x + 10)]
                        roi_mask = mask[max(0, y - 10):min(frame.shape[0], y + 10),
                                   max(0, x - 10):min(frame.shape[1], x + 10)]
                        if roi.size > 0 and roi_mask.any():
                            avg_color = np.mean(roi[roi_mask > 0], axis=0).tolist()
                            colors.append(avg_color)
            return np.mean(colors, axis=0) if colors else [0, 0, 0]

        chest_back_color = get_average_color([keypoints[i] for i in chest_back_indices])
        legs_color = get_average_color([keypoints[i] for i in legs_indices])
        arms_color = get_average_color([keypoints[i] for i in arms_indices])

        return {
            'chest_back': chest_back_color,
            'legs': legs_color,
            'arms': arms_color
        }

    def process_frame(self, frame, frame_num):
        composite_mask_file = os.path.join(self.mask_dir, f'mask_{frame_num:06d}.png')
        frame_detections = []
        frame_masks = []
        frame_colors = []
        frame_keypoints = []

        if frame_num < len(self.detections):
            frame_detections = self.detections[frame_num]['detections']
            frame_keypoints = self.detections[frame_num]['keypoints']
        else:
            # YOLO detection and pose estimation
            results = self.yolo_model(frame)
            for r in results:
                for box, kps in zip(r.boxes.data, r.keypoints.data):
                    x1, y1, x2, y2, score, class_id = box.tolist()
                    if class_id == 0:  # Ensure it's a person
                        frame_detections.append([x1, y1, x2, y2])
                        frame_keypoints.append(kps.tolist())
            self.detections.append({'detections': frame_detections, 'keypoints': frame_keypoints})

        # SAM segmentation
        self.sam_predictor.set_image(frame)
        composite_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for i, (box, keypoints) in enumerate(zip(frame_detections, frame_keypoints)):
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

            # Calculate body part colors
            body_part_colors = self.get_body_part_colors(frame, mask, keypoints)

            frame_masks.append(individual_mask_file)
            frame_colors.append(body_part_colors)

        # Save composite mask
        cv2.imwrite(composite_mask_file, composite_mask * 255)

        # Update pose-mask associations
        self.pose_mask_associations[frame_num] = {
            'detections': frame_detections,
            'masks': frame_masks,
            'colors': frame_colors,
            'keypoints': frame_keypoints
        }

        # Invert mask to get background
        bg_mask = cv2.bitwise_not(composite_mask * 255)

        # Apply mask to original frame
        bg_only = cv2.bitwise_and(frame, frame, mask=bg_mask)

        return bg_only, composite_mask

    def process_video(self, input_path, force_reprocess=False):
        if not force_reprocess and self.load_existing_data():
            print("Loaded existing detection data.")
            if os.path.exists(self.bg_video_path) and os.path.exists(self.reid_video_path):
                print(f"Background and ReID videos already exist.")
                return True
            else:
                print("Existing detection data found, but videos are missing. Reprocessing video...")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open the input video at {input_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_bg = cv2.VideoWriter(self.bg_video_path, fourcc, fps, (width, height))
        out_reid = cv2.VideoWriter(self.reid_video_path, fourcc, fps, (width, height))

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bg_only, mask = self.process_frame(frame, frame_num)
            out_bg.write(bg_only)

            frame_num += 1

        cap.release()
        out_bg.release()

        # Perform ReID
        self.perform_reid()

        # Render ReID video
        cap = cv2.VideoCapture(input_path)
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            reid_frame = self.render_reid_frame(frame, frame_num)
            out_reid.write(reid_frame)

            frame_num += 1

        cap.release()
        out_reid.release()
        cv2.destroyAllWindows()

        # Save all data
        self.save_data()

        print(f"Processed {frame_num} frames")
        print(f"Masks saved in: {self.mask_dir}")
        print(f"Detections saved in: {self.detection_file}")
        print(f"Pose-mask associations saved in: {self.pose_mask_file}")
        print(f"Person ReID data saved in: {self.person_reid_file}")
        print(f"Background video saved to: {self.bg_video_path}")
        print(f"ReID video saved to: {self.reid_video_path}")
        return True

    def perform_reid(self):
        all_chest_back_colors = []
        all_legs_colors = []
        all_arms_colors = []
        all_positions = []
        all_sizes = []
        frame_indices = []
        detection_indices = []

        for frame_num, frame_data in self.pose_mask_associations.items():
            for i, (detection, colors) in enumerate(zip(frame_data['detections'], frame_data['colors'])):
                all_chest_back_colors.append(colors['chest_back'])
                all_legs_colors.append(colors['legs'])
                all_arms_colors.append(colors['arms'])
                center_x = (detection[0] + detection[2]) / 2
                center_y = (detection[1] + detection[3]) / 2
                all_positions.append([center_x, center_y])
                size = (detection[2] - detection[0]) * (detection[3] - detection[1])
                all_sizes.append(size)
                frame_indices.append(frame_num)
                detection_indices.append(i)

        all_chest_back_colors = np.array(all_chest_back_colors)
        all_legs_colors = np.array(all_legs_colors)
        all_arms_colors = np.array(all_arms_colors)
        all_positions = np.array(all_positions)
        all_sizes = np.array(all_sizes)

        # Normalize features
        all_chest_back_colors /= 255.0
        all_legs_colors /= 255.0
        all_arms_colors /= 255.0
        all_positions /= np.max(all_positions)
        all_sizes /= np.max(all_sizes)

        # Combine features with different weights
        features = np.hstack([
            all_chest_back_colors * 0.5,
            all_legs_colors * 0.3,
            all_arms_colors * 0.2,
            all_positions * 0.1,
            all_sizes.reshape(-1, 1) * 0.1
        ])

        # Perform clustering
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

    def render_reid_frame(self, frame, frame_num):
        if frame_num not in self.pose_mask_associations:
            return frame

        frame_data = self.pose_mask_associations[frame_num]

        for person_id, person_data in self.person_reid.items():
            if frame_num in person_data:
                for detection_index in person_data[frame_num]:
                    detection = frame_data['detections'][detection_index]
                    keypoints = frame_data['keypoints'][detection_index]
                    colors = frame_data['colors'][detection_index]

                    # Draw bounding box
                    cv2.rectangle(frame, (int(detection[0]), int(detection[1])),
                                  (int(detection[2]), int(detection[3])), (0, 255, 0), 2)

                    # Draw ReID label
                    cv2.putText(frame, person_id, (int(detection[0]), int(detection[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Draw pose connections
                    self.draw_pose_connections(frame, keypoints, colors)

        return frame

    @staticmethod
    def draw_pose_connections(frame, keypoints, colors):
        connections = [
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 6), (5, 11), (6, 12), (11, 12),  # Chest/Back
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        for connection in connections:
            start_point = keypoints[connection[0]]
            end_point = keypoints[connection[1]]

            if start_point[2] > 0.5 and end_point[2] > 0.5:
                start_point = (int(start_point[0]), int(start_point[1]))
                end_point = (int(end_point[0]), int(end_point[1]))

                if connection[0] in [5, 6, 11, 12] and connection[1] in [5, 6, 11, 12]:
                    color = colors['chest_back']
                elif connection[0] in [11, 12, 13, 14, 15, 16] and connection[1] in [11, 12, 13, 14, 15, 16]:
                    color = colors['legs']
                else:
                    color = colors['arms']

                cv2.line(frame, start_point, end_point, color, 2)

    def numpy_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.numpy_to_list(item) for item in obj]
        else:
            return obj

    def save_data(self):
        with open(self.detection_file, 'w') as f:
            json.dump(self.numpy_to_list(self.detections), f, indent=2)

        with open(self.pose_mask_file, 'w') as f:
            json.dump(self.numpy_to_list(self.pose_mask_associations), f, indent=2)

        with open(self.person_reid_file, 'w') as f:
            json.dump(self.numpy_to_list(self.person_reid), f, indent=2)

    def get_bg_video_path(self):
        return self.bg_video_path

    def get_reid_video_path(self):
        return self.reid_video_path
