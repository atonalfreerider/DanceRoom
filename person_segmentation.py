import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import json
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter


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
        self.kalman_filters = {}
        self.person_trackers = {}
        self.max_person_id = 0

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

        # YOLO detection and pose estimation
        results = self.yolo_model(frame)
        for r in results:
            for box, kps in zip(r.boxes.data, r.keypoints.data):
                x1, y1, x2, y2, score, class_id = box.tolist()
                if class_id == 0:  # Ensure it's a person
                    frame_detections.append([x1, y1, x2, y2])
                    frame_keypoints.append(kps.tolist())

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

        # Perform ReID immediately after detection
        self.perform_reid_for_frame(frame_num)

        return bg_only, composite_mask

    def perform_reid_for_frame(self, frame_num):
        if frame_num == 0:
            # For the first frame, assign new IDs to all detections
            for i in range(len(self.pose_mask_associations[0]['detections'])):
                self.create_new_person(0, i)
        else:
            # For subsequent frames, use Kalman filter to predict and match
            self.predict_and_match(frame_num)

    def create_new_person(self, frame_num, detection_index):
        self.max_person_id += 1
        person_id = f"person_{self.max_person_id}"
        detection = self.pose_mask_associations[frame_num]['detections'][detection_index]

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.R *= 10
        kf.Q *= 0.1

        center_x = (detection[0] + detection[2]) / 2
        center_y = (detection[1] + detection[3]) / 2
        kf.x = np.array([center_x, center_y, 0, 0])

        self.person_trackers[person_id] = {
            'kf': kf,
            'last_seen': frame_num,
            'color_history': [self.pose_mask_associations[frame_num]['colors'][detection_index]]
        }

        if person_id not in self.person_reid:
            self.person_reid[person_id] = {}
        self.person_reid[person_id][frame_num] = [detection_index]

    def predict_and_match(self, frame_num):
        current_detections = self.pose_mask_associations[frame_num]['detections']
        detection_centers = np.array([(d[0] + d[2]) / 2 for d in current_detections])
        detection_centers = np.column_stack((detection_centers, [(d[1] + d[3]) / 2 for d in current_detections]))

        predictions = {}
        for person_id, tracker in self.person_trackers.items():
            tracker['kf'].predict()
            predictions[person_id] = tracker['kf'].x[:2]

        cost_matrix = cdist(np.array(list(predictions.values())), detection_centers)

        matched_indices = linear_sum_assignment(cost_matrix)[1]

        for i, detection_idx in enumerate(matched_indices):
            person_id = list(predictions.keys())[i]
            if cost_matrix[i, detection_idx] < 100:  # Threshold for matching
                self.update_person(person_id, frame_num, detection_idx)
            else:
                self.create_new_person(frame_num, detection_idx)

        # Handle unmatched detections
        unmatched_detections = set(range(len(current_detections))) - set(matched_indices)
        for detection_idx in unmatched_detections:
            self.create_new_person(frame_num, detection_idx)

    def update_person(self, person_id, frame_num, detection_idx):
        detection = self.pose_mask_associations[frame_num]['detections'][detection_idx]
        center_x = (detection[0] + detection[2]) / 2
        center_y = (detection[1] + detection[3]) / 2

        self.person_trackers[person_id]['kf'].update(np.array([center_x, center_y]))
        self.person_trackers[person_id]['last_seen'] = frame_num
        self.person_trackers[person_id]['color_history'].append(
            self.pose_mask_associations[frame_num]['colors'][detection_idx]
        )

        if person_id not in self.person_reid:
            self.person_reid[person_id] = {}
        self.person_reid[person_id][frame_num] = [detection_idx]

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

        frame_num = 0
        self.detections = []
        self.pose_mask_associations = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bg_only, mask = self.process_frame(frame, frame_num)
            out_bg.write(bg_only)

            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames")

        cap.release()
        out_bg.release()

        print("Performing final ReID validation...")
        self.validate_reid()

        print("Saving data...")
        self.save_data()

        self.generate_reid_video(input_path)

        print(f"Processing complete.")
        print(f"Processed {frame_num} frames")
        print(f"Background video saved to: {self.bg_video_path}")
        print(f"ReID video saved to: {self.reid_video_path}")
        return True

    def generate_reid_video(self, input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open the input video at {input_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.reid_video_path, fourcc, fps, (width, height))

        for frame_num in range(len(self.detections)):
            ret, frame = cap.read()
            if not ret:
                break

            reid_frame = self.render_reid_frame(frame, frame_num)
            out.write(reid_frame)

        cap.release()
        out.release()

    def validate_reid(self):
        color_features = {}
        for person_id, tracker in self.person_trackers.items():
            color_history = tracker['color_history']
            avg_color = {
                'chest_back': np.mean([c['chest_back'] for c in color_history], axis=0),
                'legs': np.mean([c['legs'] for c in color_history], axis=0),
                'arms': np.mean([c['arms'] for c in color_history], axis=0)
            }
            color_features[person_id] = np.concatenate([
                avg_color['chest_back'],
                avg_color['legs'],
                avg_color['arms']
            ])

        feature_matrix = np.array(list(color_features.values()))
        distance_matrix = cdist(feature_matrix, feature_matrix)

        merged_ids = set()
        for i in range(len(feature_matrix)):
            if i in merged_ids:
                continue
            similar_ids = np.where(distance_matrix[i] < 0.1)[0]  # Adjust threshold as needed
            for j in similar_ids:
                if i != j and j not in merged_ids:
                    self.merge_person_ids(list(color_features.keys())[i], list(color_features.keys())[j])
                    merged_ids.add(j)

    def merge_person_ids(self, keep_id, merge_id):
        for frame, detections in self.person_reid[merge_id].items():
            if frame not in self.person_reid[keep_id]:
                self.person_reid[keep_id][frame] = detections
            else:
                self.person_reid[keep_id][frame].extend(detections)
        del self.person_reid[merge_id]
        del self.person_trackers[merge_id]

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
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {self.numpy_to_list(key): self.numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.numpy_to_list(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.numpy_to_list(item) for item in obj)
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
