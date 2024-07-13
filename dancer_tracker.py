import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import matplotlib.pyplot as plt
from yolox.tracker.byte_tracker import BYTETracker
import torch
import tqdm


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class DancerTracker:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, 'depth')

        self.men_women_file = os.path.join(output_dir, 'men-women.json')
        self.detections_file = os.path.join(output_dir, 'detections.json')
        self.lead_file = os.path.join(output_dir, 'lead.json')
        self.follow_file = os.path.join(output_dir, 'follow.json')

        self.men_women = self.load_json(self.men_women_file)
        self.detections = self.load_json(self.detections_file)
        self.lead = self.load_json(self.lead_file)
        self.follow = self.load_json(self.follow_file)

    def process_video(self):
        self.detect_men_women()
        self.match_poses_and_identify_leads()
        self.generate_debug_video()
        print("Video processing complete.")

    def detect_men_women(self):
        if self.men_women:
            print("Using cached men-women detections.")
            return

        model = YOLO('yolov8x-man-woman.pt')
        cap = cv2.VideoCapture(self.input_path)
        frame_count = 0

        pbar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Detecting men and women")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            men = []
            women = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    cls = box.cls.item()
                    if cls == 0:  # Assuming 0 is the class for men
                        men.append([x1, y1, x2, y2, conf])
                    elif cls == 1:  # Assuming 1 is the class for women
                        women.append([x1, y1, x2, y2, conf])

            self.men_women[frame_count] = {'men': men, 'women': women}
            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        self.save_json(self.men_women, self.men_women_file)
        print(f"Saved men-women detections for {frame_count} frames.")

    def match_poses_and_identify_leads(self):
        cap = cv2.VideoCapture(self.input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        tracker = BYTETracker(BYTETrackerArgs())
        tracked_sequences = {}

        for frame_num in self.men_women.keys():
            depth_map = self.load_depth_map(int(frame_num))
            if depth_map is None:
                continue

            men = self.men_women[frame_num]['men']
            women = self.men_women[frame_num]['women']
            poses = self.detections.get(frame_num, [])

            # Calculate scores, depths, sizes, and genders for all poses
            pose_data = []
            for pose in poses:
                keypoints = pose["keypoints"]
                avg_depth, size = self.calculate_pose_score(keypoints, depth_map, frame_width, frame_height)
                man_conf = max([box[4] for box in men if self.pose_in_box(keypoints, box)], default=0)
                woman_conf = max([box[4] for box in women if self.pose_in_box(keypoints, box)], default=0)
                gender = 'man' if man_conf > woman_conf else 'woman'
                gender_conf = max(man_conf, woman_conf)
                pose_size = self.calculate_pose_size(keypoints)
                pose_data.append((keypoints, avg_depth, size, gender, gender_conf, pose_size))

            # Sort poses by depth (closest first)
            pose_data.sort(key=lambda x: x[1])

            # Get the two closest poses
            closest_poses = pose_data[:2] if len(pose_data) >= 2 else pose_data

            # Prepare detections for ByteTracker
            dets = []
            for p in closest_poses:
                bbox = self.pose_to_bbox(p[0])
                confidence = p[4]  # This is the gender confidence
                class_score = 1.0  # Assuming all detections are of the same class (person)
                dets.append(bbox + [confidence, class_score])

            dets = torch.tensor(dets, dtype=torch.float32)

            if len(dets) > 0:
                img_info = [frame_height, frame_width]  # Changed to list
                img_size = [frame_height, frame_width]
                online_targets = tracker.update(dets, img_info, img_size)

                # Update tracked sequences
                for t in online_targets:
                    track_id = t.track_id
                    if track_id not in tracked_sequences:
                        tracked_sequences[track_id] = []

                    # Find the closest pose to the tracked object
                    tlwh = t.tlwh
                    center_x, center_y = tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2
                    closest_pose = min(closest_poses, key=lambda p: self.distance_to_center(p[0], center_x, center_y))

                    tracked_sequences[track_id].append((frame_num, closest_pose))
            else:
                print(f"No detections for frame {frame_num}")

        # Assign genders to tracked sequences
        for track_id, sequence in tracked_sequences.items():
            gender_votes = {'man': 0, 'woman': 0}
            for _, pose_data in sequence:
                gender_votes[pose_data[3]] += pose_data[4]  # Add gender confidence
            final_gender = max(gender_votes, key=gender_votes.get)

    def pose_to_bbox(self, pose):
        valid_points = [p[:2] for p in pose if p[2] > 0]
        if not valid_points:
            return [0, 0, 1, 1]  # Return a small box if no valid points
        x_coords, y_coords = zip(*valid_points)
        left, top = min(x_coords), min(y_coords)
        right, bottom = max(x_coords), max(y_coords)
        return [left, top, right, bottom]

    def distance_to_center(self, pose, center_x, center_y):
        valid_points = [p[:2] for p in pose if p[2] > 0]
        if not valid_points:
            return float('inf')
        pose_center_x = sum(p[0] for p in valid_points) / len(valid_points)
        pose_center_y = sum(p[1] for p in valid_points) / len(valid_points)
        return ((pose_center_x - center_x) ** 2 + (pose_center_y - center_y) ** 2) ** 0.5

    @staticmethod
    def calculate_pose_size(pose):
        head_keypoints = [0, 1, 2, 3, 4]  # Nose, left eye, right eye, left ear, right ear
        foot_keypoints = [15, 16]  # Left ankle, right ankle

        head_y = min([pose[i][1] for i in head_keypoints if i < len(pose) and len(pose[i]) > 2 and pose[i][2] > 0],
                     default=None)
        foot_y = max([pose[i][1] for i in foot_keypoints if i < len(pose) and len(pose[i]) > 2 and pose[i][2] > 0],
                     default=None)

        if head_y is not None and foot_y is not None:
            return foot_y - head_y
        else:
            return 0  # Return 0 if we can't calculate the size

    @staticmethod
    def calculate_pose_score(pose, depth_map, frame_width, frame_height):
        pose_depths = []
        valid_keypoints = []
        for kp in pose:
            x, y, conf = kp[0], kp[1], kp[2]
            if conf > 0 and x > 0 and y > 0:
                x_scaled = int(x * 640 / frame_width)
                y_scaled = int(y * 480 / frame_height)
                if 0 <= x_scaled < 640 and 0 <= y_scaled < 480:
                    pose_depths.append(depth_map[y_scaled, x_scaled])
                    valid_keypoints.append((x, y))

        if not pose_depths:
            return float('inf'), 0

        avg_depth = np.mean(pose_depths)

        # Calculate pose size (bounding box area)
        if valid_keypoints:
            x_coords, y_coords = zip(*valid_keypoints)
            size = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        else:
            size = 0

        return avg_depth, size

    @staticmethod
    def pose_in_box(pose, box):
        valid_points = [p for p in pose if p[2] > 0]  # Consider only points with confidence > 0
        if not valid_points:
            return False
        x_coords, y_coords, _ = zip(*valid_points)
        pose_center_x = sum(x_coords) / len(x_coords)
        pose_center_y = sum(y_coords) / len(y_coords)
        return box[0] <= pose_center_x <= box[2] and box[1] <= pose_center_y <= box[3]

    @staticmethod
    def get_3d_keypoints(pose, depth_map, frame_width, frame_height):
        keypoints_3d = []
        for kp in pose:
            x, y, conf = kp
            if x == 0 and y == 0:  # Disregard [0, 0] points
                keypoints_3d.append([x, y, 0, 0])  # Add a zero-confidence point
                continue
            x_scaled = int(x * 640 / frame_width)
            y_scaled = int(y * 480 / frame_height)
            if 0 <= x_scaled < 640 and 0 <= y_scaled < 480:
                z = float(depth_map[y_scaled, x_scaled])  # Convert to float
                keypoints_3d.append([float(x), float(y), z, float(conf)])  # Convert all to float
            else:
                keypoints_3d.append([float(x), float(y), 0, float(conf)])  # Convert all to float
        return keypoints_3d

    #region UTILITY

    @staticmethod
    def load_json(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}

    def load_depth_map(self, frame_num):
        depth_file = os.path.join(self.depth_dir, f'{frame_num:06d}.npz')
        if os.path.exists(depth_file):
            with np.load(depth_file) as data:
                # Try to get the first key in the archive
                keys = list(data.keys())
                if keys:
                    return data[keys[0]]
                else:
                    print(f"Warning: No data found in {depth_file}")
                    return None
        else:
            print(f"Warning: Depth file not found: {depth_file}")
            return None

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.numpy_to_python(data), f, indent=2)

    #endregion

    #region DEBUG

    def generate_debug_video(self):
        debug_video_path = os.path.join(self.output_dir, 'debug_video.mp4')
        cap = cv2.VideoCapture(self.input_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(debug_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            depth_map = self.load_depth_map(frame_count)
            if depth_map is None:
                print(f"Skipping frame {frame_count} due to missing depth map")
                frame_count += 1
                continue

            depth_pred_col = self.colorize(depth_map, vmin=0.01, vmax=10.0, cmap="magma_r")
            depth_pred_col = cv2.resize(depth_pred_col, (frame_width, frame_height))
            debug_frame = cv2.cvtColor(depth_pred_col, cv2.COLOR_RGB2BGR)

            poses = self.detections.get(str(frame_count), [])
            for pose in poses:
                self.draw_pose(debug_frame, pose["keypoints"], (128, 128, 128))  # Gray for all poses

            lead_pose = self.lead.get(str(frame_count))
            if lead_pose:
                self.draw_pose(debug_frame, lead_pose, (255, 0, 0), is_lead_or_follow=True)  # Blue for lead man

            follow_pose = self.follow.get(str(frame_count))
            if follow_pose:
                self.draw_pose(debug_frame, follow_pose, (0, 255, 0), is_lead_or_follow=True)  # Green for follow woman

            out.write(debug_frame)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames for debug video")

        cap.release()
        out.release()

        if frame_count == 0:
            print("Error: No frames were processed. Check your input video and depth maps.")
        else:
            print(f"Debug video saved to {debug_video_path} with {frame_count} frames")

    @staticmethod
    def draw_pose(image, pose, color, is_lead_or_follow=False):
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]

        overlay = np.zeros_like(image, dtype=np.uint8)

        def get_point_and_conf(kp):
            if len(kp) == 4 and (kp[0] != 0 or kp[1] != 0):
                return (int(kp[0]), int(kp[1])), kp[3]  # x, y, z, conf
            elif len(kp) == 3 and (kp[0] != 0 or kp[1] != 0):
                return (int(kp[0]), int(kp[1])), kp[2]  # x, y, conf
            return None, 0.0

        line_thickness = 3 if is_lead_or_follow else 1

        for connection in connections:
            if len(pose) > max(connection):
                start_point, start_conf = get_point_and_conf(pose[connection[0]])
                end_point, end_conf = get_point_and_conf(pose[connection[1]])

                if start_point is not None and end_point is not None:
                    avg_conf = (start_conf + end_conf) / 2
                    color_with_alpha = tuple(int(c * avg_conf) for c in color)
                    cv2.line(overlay, start_point, end_point, color_with_alpha, line_thickness)

        for point in pose:
            pt, conf = get_point_and_conf(point)
            if pt is not None:
                color_with_alpha = tuple(int(c * conf) for c in color)
                cv2.circle(overlay, pt, 3, color_with_alpha, -1)

        cv2.add(image, overlay, image)

    @staticmethod
    def colorize(value, vmin=None, vmax=None, cmap="magma_r"):
        if value.ndim > 2:
            if value.shape[-1] > 1:
                return value
            value = value[..., 0]
        invalid_mask = value < 0.0001
        vmin = value.min() if vmin is None else vmin
        vmax = value.max() if vmax is None else vmax
        value = (value - vmin) / (vmax - vmin)
        cmapper = plt.get_cmap(cmap)
        value = cmapper(value, bytes=True)
        value[invalid_mask] = 0
        img = value[..., :3]
        return img

    def numpy_to_python(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self.numpy_to_python(obj.tolist())
        elif isinstance(obj, list):
            return [self.numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.numpy_to_python(value) for key, value in obj.items()}
        else:
            return obj

    #endregion
