import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import json
import os
import glob
import matplotlib.pyplot as plt


class DanceSegmentation:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.mask_dir = os.path.join(output_dir, 'masks')
        self.depth_dir = os.path.join(output_dir, 'depth')
        os.makedirs(self.mask_dir, exist_ok=True)

        self.men_women_file = os.path.join(output_dir, 'men-women.json')
        self.detections_file = os.path.join(output_dir, 'detections.json')
        self.lead_file = os.path.join(output_dir, 'lead.json')
        self.follow_file = os.path.join(output_dir, 'follow.json')

        self.men_women = self.load_json(self.men_women_file)
        self.detections = self.load_json(self.detections_file)
        self.lead = self.load_json(self.lead_file)
        self.follow = self.load_json(self.follow_file)

        self.sam_predictor = None

    def process_video(self):
        self.detect_men_women()
        self.detect_poses()
        self.match_poses_and_identify_leads()
        # self.segment_leads()
        self.generate_debug_video()
        print("Video processing complete.")

    def detect_men_women(self):
        if self.men_women:
            print("Using cached men-women detections.")
            return

        model = YOLO('yolov8x-man-woman.pt')
        cap = cv2.VideoCapture(self.input_path)
        frame_count = 0

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

        cap.release()
        self.save_json(self.men_women, self.men_women_file)
        print(f"Saved men-women detections for {frame_count} frames.")

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
            poses = []

            for r in results:
                for pose in r.keypoints.data:
                    poses.append(pose.tolist())

            self.detections[frame_count] = poses
            frame_count += 1

        cap.release()
        self.save_json(self.detections, self.detections_file)
        print(f"Saved pose detections for {frame_count} frames.")

    def match_poses_and_identify_leads(self):
        if self.lead and self.follow:
            print("Using cached lead and follow data.")
            return

        cap = cv2.VideoCapture(self.input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

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
                score, avg_depth, size = self.calculate_pose_score(pose, depth_map, frame_width, frame_height)
                gender = 'man' if self.pose_in_boxes(pose, men) else 'woman' if self.pose_in_boxes(pose,
                                                                                                   women) else 'neutral'
                pose_size = self.calculate_pose_size(pose)
                pose_data.append((pose, score, avg_depth, size, gender, pose_size))

            # Sort poses by depth (closest first)
            pose_data.sort(key=lambda x: x[2])

            # Get the two closest poses
            closest_poses = pose_data[:2] if len(pose_data) >= 2 else pose_data

            lead = None
            follow = None

            if len(closest_poses) == 2:
                if closest_poses[0][4] == 'man' and closest_poses[1][4] == 'man':
                    # Two closest are men, larger is lead, smaller is follow
                    if closest_poses[0][5] > closest_poses[1][5]:
                        lead, follow = closest_poses[0], closest_poses[1]
                    else:
                        lead, follow = closest_poses[1], closest_poses[0]
                elif closest_poses[0][4] == 'woman' and closest_poses[1][4] == 'woman':
                    # Two closest are women, larger is lead, smaller is follow
                    if closest_poses[0][5] > closest_poses[1][5]:
                        lead, follow = closest_poses[0], closest_poses[1]
                    else:
                        lead, follow = closest_poses[1], closest_poses[0]
                else:
                    # One man and one woman, or other combinations
                    lead = next((p for p in closest_poses if p[4] == 'man'), None)
                    follow = next((p for p in closest_poses if p[4] == 'woman'), None)

                    # If we don't have both lead and follow, assign based on depth
                    if not lead and not follow:
                        lead, follow = closest_poses[0], closest_poses[1]
                    elif not lead:
                        lead = [p for p in closest_poses if p != follow][0]
                    elif not follow:
                        follow = [p for p in closest_poses if p != lead][0]
            elif len(closest_poses) == 1:
                lead = closest_poses[0]

            # Convert to 3D keypoints
            if lead:
                self.lead[frame_num] = self.get_3d_keypoints(lead[0], depth_map, frame_width, frame_height)
            if follow:
                self.follow[frame_num] = self.get_3d_keypoints(follow[0], depth_map, frame_width, frame_height)

        self.save_json(self.lead, self.lead_file)
        self.save_json(self.follow, self.follow_file)
        print("Saved lead and follow data.")

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
        pose_confidences = []
        valid_keypoints = []
        for kp in pose:
            x, y, conf = kp[0], kp[1], kp[2]
            if conf > 0:
                x_scaled = int(x * 640 / frame_width)
                y_scaled = int(y * 480 / frame_height)
                if 0 <= x_scaled < 640 and 0 <= y_scaled < 480:
                    pose_depths.append(depth_map[y_scaled, x_scaled])
                    pose_confidences.append(conf)
                    valid_keypoints.append((x, y))

        if not pose_depths:
            return 0, float('inf'), 0

        avg_depth = np.mean(pose_depths)
        avg_confidence = np.mean(pose_confidences)

        # Calculate pose size (bounding box area)
        if valid_keypoints:
            x_coords, y_coords = zip(*valid_keypoints)
            size = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        else:
            size = 0

        # Calculate a score that favors closer poses with higher confidence
        score = avg_confidence / (avg_depth + 1e-6)  # Add small epsilon to avoid division by zero
        return score, avg_depth, size

    @staticmethod
    def pose_in_boxes(pose, boxes):
        valid_points = [p for p in pose if p[2] > 0]  # Consider only points with confidence > 0
        if not valid_points:
            return False
        x_coords, y_coords, _ = zip(*valid_points)
        pose_center_x = sum(x_coords) / len(x_coords)
        pose_center_y = sum(y_coords) / len(y_coords)
        return any(box[0] <= pose_center_x <= box[2] and box[1] <= pose_center_y <= box[3] for box in boxes)

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

    def segment_leads(self):
        if not self.sam_predictor:
            sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
            self.sam_predictor = SamPredictor(sam)

        cap = cv2.VideoCapture(self.input_path)

        existing_masks = glob.glob(os.path.join(self.mask_dir, 'mask_lead_??????.png'))
        start_frame = len(existing_masks)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            lead_pose = self.lead.get(str(frame_count))
            follow_pose = self.follow.get(str(frame_count))

            if lead_pose:
                self.segment_person(frame, lead_pose, frame_count, 'lead')
            if follow_pose:
                self.segment_person(frame, follow_pose, frame_count, 'follow')

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames for segmentation")

        cap.release()

    def segment_person(self, frame, pose, frame_num, person_type):
        self.sam_predictor.set_image(frame)

        # Get bounding box from pose
        x_coords = [kp[0] for kp in pose if kp[3] > 0.5]  # Only consider high confidence keypoints
        y_coords = [kp[1] for kp in pose if kp[3] > 0.5]
        if not x_coords or not y_coords:
            return

        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        box = np.array([x1, y1, x2, y2])
        masks, _, _ = self.sam_predictor.predict(box=box, multimask_output=False)
        mask = masks[0].astype(np.uint8)

        mask_file = os.path.join(self.mask_dir, f'mask_{person_type}_{frame_num:06d}.png')
        cv2.imwrite(mask_file, mask * 255)

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
                self.draw_pose(debug_frame, pose, (128, 128, 128))  # Gray for all poses

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
