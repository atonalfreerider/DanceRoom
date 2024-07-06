import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import json
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
import glob


class PersonSegmentation:
    def __init__(self, output_dir):
        self.sam_predictor = None
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
        self.load_cached_data()
        self.max_age = 60
        self.min_hits = 3
        self.max_persons = 8

    #region INITIALIZATION

    @staticmethod
    def initialize_sam():
        sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
        return SamPredictor(sam)

    def load_cached_data(self):
        if os.path.exists(self.detection_file):
            with open(self.detection_file, 'r') as f:
                self.detections = json.load(f)
            print(f"Loaded {len(self.detections)} cached detections.")
        else:
            print("No cached detections found.")

        if os.path.exists(self.pose_mask_file):
            with open(self.pose_mask_file, 'r') as f:
                self.pose_mask_associations = json.load(f)
            print(f"Loaded pose-mask associations for {len(self.pose_mask_associations)} frames.")
        else:
            print("No cached pose-mask associations found.")

        if os.path.exists(self.person_reid_file):
            with open(self.person_reid_file, 'r') as f:
                self.person_reid = json.load(f)
            print(f"Loaded person ReID data for {len(self.person_reid)} persons.")
        else:
            print("No cached person ReID data found.")

    #endregion

    #region VIDEO LOOP

    def process_video(self, input_path, force_reprocess=False):
        # Step 1: Perform detections
        self.perform_detections(input_path, force_reprocess)

        # Step 2: Perform segmentation
        self.perform_segmentation(input_path, force_reprocess)

        if not self.person_reid or force_reprocess:
            print("Performing ReID...")
            self.perform_reid()
            self.save_data()
        else:
            print("Using cached person ReID data.")

        print("Generating ReID video...")
        self.generate_reid_video(input_path)

        print(f"Processing complete.")
        print(f"Background video saved to: {self.bg_video_path}")
        print(f"ReID video saved to: {self.reid_video_path}")
        return True

    def perform_detections(self, input_path, force_reprocess=False):
        if not force_reprocess and os.path.exists(self.detection_file):
            print("Using cached detections.")
            with open(self.detection_file, 'r') as f:
                self.detections = json.load(f)
            return

        yolo_model = YOLO('yolov8x-pose-p6.pt')
        self.detections = []
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open the input video at {input_path}")
            return False

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)
            frame_detections = []
            frame_keypoints = []
            for r in results:
                for box, kps in zip(r.boxes.data, r.keypoints.data):
                    x1, y1, x2, y2, score, class_id = box.tolist()
                    if class_id == 0:  # Ensure it's a person
                        frame_detections.append([x1, y1, x2, y2])
                        frame_keypoints.append(kps.tolist())

            self.detections.append({
                'detections': frame_detections,
                'keypoints': frame_keypoints
            })

            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames for detection")

        cap.release()

        # Save detections
        with open(self.detection_file, 'w') as f:
            json.dump(self.detections, f, indent=2)
        print(f"Saved {len(self.detections)} detections to {self.detection_file}")

    def perform_segmentation(self, input_path, force_reprocess=False):
        # Determine the starting frame
        existing_masks = glob.glob(os.path.join(self.mask_dir, 'mask_??????.png'))
        start_frame = len(existing_masks)

        # if start_frame is the last frame of the video, then we are done
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not force_reprocess and start_frame == total_frames:
            print("Segmentation already completed for all frames.")
            return

        if not force_reprocess and start_frame > 0:
            print(f"Continuing segmentation from frame {start_frame}")
        else:
            start_frame = 0
            self.pose_mask_associations = {}

        self.sam_predictor = self.initialize_sam()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open the input video at {input_path}")
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_num = start_frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.segment_frame_and_match(frame, frame_num)

            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames for segmentation")

        cap.release()

        # Save pose-mask associations
        self.save_data()

    def segment_frame_and_match(self, frame, frame_num):
        composite_mask_file = os.path.join(self.mask_dir, f'mask_{frame_num:06d}.png')
        frame_masks = []
        frame_colors = []

        # Use pre-computed detections
        frame_detections = self.detections[frame_num]['detections']
        frame_keypoints = self.detections[frame_num]['keypoints']

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

    #endregion

    #region ReID

    def create_new_person(self, frame_num, detection_index):
        if len(self.person_trackers) >= self.max_persons:
            return

        self.max_person_id += 1
        person_id = f"person_{self.max_person_id}"
        detection = self.pose_mask_associations[frame_num]['detections'][detection_index]
        color = self.pose_mask_associations[frame_num]['colors'][detection_index]

        kf = KalmanFilter(dim_x=6, dim_z=5)  # State: [x, y, w, h, c, v]
        kf.F = np.eye(6)
        kf.F[:5, 5] = 1  # velocity component
        kf.H = np.eye(5, 6)
        kf.R *= 10
        kf.Q[-1, -1] *= 0.01  # small changes in velocity

        x, y, w, h = detection[0], detection[1], detection[2] - detection[0], detection[3] - detection[1]
        c = np.mean([np.mean(color['chest_back']), np.mean(color['legs']), np.mean(color['arms'])])
        kf.x = np.array([x, y, w, h, c, 0])

        self.person_trackers[person_id] = {
            'kf': kf,
            'last_seen': frame_num,
            'color_history': [color],
            'age': 0,
            'hits': 1
        }

        if person_id not in self.person_reid:
            self.person_reid[person_id] = {}
        self.person_reid[person_id][frame_num] = [detection_index]

    def perform_reid_for_frame(self, frame_num):
        if frame_num == 0:
            for i in range(len(self.pose_mask_associations[0]['detections'])):
                self.create_new_person(0, i)
        else:
            self.predict_and_match(frame_num)

    def predict_and_match(self, frame_num):
        current_detections = self.pose_mask_associations[frame_num]['detections']
        current_colors = self.pose_mask_associations[frame_num]['colors']

        detection_features = []
        for d, c in zip(current_detections, current_colors):
            x, y, w, h = d[0], d[1], d[2] - d[0], d[3] - d[1]
            color = np.mean([np.mean(c['chest_back']), np.mean(c['legs']), np.mean(c['arms'])])
            detection_features.append([x, y, w, h, color])
        detection_features = np.array(detection_features)

        predictions = {}
        for person_id, tracker in self.person_trackers.items():
            tracker['kf'].predict()
            predictions[person_id] = tracker['kf'].x[:5]  # x, y, w, h, color

        if not predictions:
            for detection_idx in range(len(current_detections)):
                self.create_new_person(frame_num, detection_idx)
            return

        cost_matrix = cdist(np.array(list(predictions.values())), detection_features)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_person_ids = set()
        for i, detection_idx in zip(row_ind, col_ind):
            person_id = list(predictions.keys())[i]
            if cost_matrix[i, detection_idx] < 100:  # Adjust this threshold as needed
                self.update_person(person_id, frame_num, detection_idx)
                matched_person_ids.add(person_id)
            else:
                if len(self.person_trackers) < self.max_persons:
                    self.create_new_person(frame_num, detection_idx)

        # Update age for all trackers and remove old ones
        for person_id in list(self.person_trackers.keys()):
            if person_id not in matched_person_ids:
                self.person_trackers[person_id]['age'] += 1
                if self.person_trackers[person_id]['age'] > self.max_age:
                    del self.person_trackers[person_id]
            else:
                self.person_trackers[person_id]['age'] = 0
                self.person_trackers[person_id]['hits'] += 1

        # Remove trackers that haven't been matched enough times
        self.person_trackers = {pid: tracker for pid, tracker in self.person_trackers.items()
                                if tracker['hits'] >= self.min_hits}

    def update_person(self, person_id, frame_num, detection_idx):
        detection = self.pose_mask_associations[frame_num]['detections'][detection_idx]
        color = self.pose_mask_associations[frame_num]['colors'][detection_idx]

        x, y, w, h = detection[0], detection[1], detection[2] - detection[0], detection[3] - detection[1]
        c = np.mean([np.mean(color['chest_back']), np.mean(color['legs']), np.mean(color['arms'])])

        measurement = np.array([x, y, w, h, c])
        self.person_trackers[person_id]['kf'].update(measurement)

        self.person_trackers[person_id]['last_seen'] = frame_num
        self.person_trackers[person_id]['color_history'].append(color)
        self.person_trackers[person_id]['age'] = 0
        self.person_trackers[person_id]['hits'] += 1

        if person_id not in self.person_reid:
            self.person_reid[person_id] = {}
        self.person_reid[person_id][frame_num] = [detection_idx]

    def perform_reid(self):
        self.person_reid = {}
        self.person_trackers = {}
        self.max_person_id = 0

        # Initial tracking using Kalman Filter
        for frame_num, frame_data in self.pose_mask_associations.items():
            self.perform_reid_for_frame(frame_num)

        # Remove short tracks (likely false positives)
        min_track_length = 10
        self.person_reid = {pid: frames for pid, frames in self.person_reid.items() if len(frames) >= min_track_length}

        # Merge similar tracks
        self.merge_similar_tracks()

    def merge_similar_tracks(self):
        track_features = {}
        for person_id, frames in self.person_reid.items():
            colors = [self.pose_mask_associations[frame]['colors'][idx[0]] for frame, idx in frames.items()]
            avg_color = np.mean([np.mean([c['chest_back'], c['legs'], c['arms']], axis=0) for c in colors], axis=0)
            track_features[person_id] = avg_color

        merged_ids = set()
        for id1 in track_features:
            if id1 in merged_ids:
                continue
            for id2 in track_features:
                if id1 != id2 and id2 not in merged_ids:
                    if np.linalg.norm(track_features[id1] - track_features[id2]) < 30:  # Adjust threshold as needed
                        self.merge_tracks(id1, id2)
                        merged_ids.add(id2)

    def merge_tracks(self, id1, id2):
        for frame, detections in self.person_reid[id2].items():
            if frame in self.person_reid[id1]:
                self.person_reid[id1][frame].extend(detections)
            else:
                self.person_reid[id1][frame] = detections
        del self.person_reid[id2]

    #endregion

    #region RENDER

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

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            reid_frame = self.render_reid_frame(frame, frame_num)
            out.write(reid_frame)

            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Rendered {frame_num} frames")

        cap.release()
        out.release()
        print(f"ReID video generated with {frame_num} frames.")

    def render_reid_frame(self, frame, frame_num):
        if str(frame_num) not in self.pose_mask_associations:
            return frame

        frame_data = self.pose_mask_associations[str(frame_num)]

        for person_id, person_data in self.person_reid.items():
            if str(frame_num) in person_data:
                for detection_index in person_data[str(frame_num)]:
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

    #endregion

    #region REFERENCE

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
        print(f"Saved {len(self.detections)} detections to {self.detection_file}")

        with open(self.pose_mask_file, 'w') as f:
            json.dump(self.numpy_to_list(self.pose_mask_associations), f, indent=2)
        print(f"Saved pose-mask associations for {len(self.pose_mask_associations)} frames to {self.pose_mask_file}")

        with open(self.person_reid_file, 'w') as f:
            json.dump(self.numpy_to_list(self.person_reid), f, indent=2)
        print(f"Saved person ReID data to {self.person_reid_file}")

    def get_bg_video_path(self):
        return self.bg_video_path

    #endregion