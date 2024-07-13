import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
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
        self.figure_mask_dir = os.path.join(output_dir, 'figure-masks')

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
        self.track_lead_and_follow()
        self.generate_debug_video()
        print("Video processing complete.")

    def detect_men_women(self):
        if self.men_women:
            print("Using cached men-women detections.")
            return

        model = YOLO('yolov8x-man-woman.pt')

        frame_count = len(os.listdir(self.figure_mask_dir))
        pbar = tqdm.tqdm(total=frame_count, desc="Detecting men and women")

        for frame_num in range(frame_count):
            figure_mask_path = os.path.join(self.figure_mask_dir, f'{frame_num:06d}.png')

            frame = cv2.imread(figure_mask_path)
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

            self.men_women[frame_num] = {'men': men, 'women': women}
            pbar.update(1)

        pbar.close()

        self.save_json(self.men_women, self.men_women_file)
        print(f"Saved men-women detections for {frame_count} frames.")

    def track_lead_and_follow(self):
        cap = cv2.VideoCapture(self.input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        tracker = BYTETracker(BYTETrackerArgs())
        tracked_sequences = {}

        frame_count = len(os.listdir(self.figure_mask_dir))
        pbar = tqdm.tqdm(total=frame_count, desc="tracking poses")

        # first, track the poses through the frames using ByteTracker and also identify genders
        for frame_num in range(frame_count):
            detections_in_frame = self.detections.get(str(frame_num), [])
            # TODO match the gender identification with the each pose in the frame

            # convert to torch tensor
            detections_in_frame_torch = torch.tensor(detections_in_frame, dtype=torch.float32)

            if len(detections_in_frame_torch) > 0:
                img_info = [frame_height, frame_width]  # Changed to list
                img_size = [frame_height, frame_width]
                online_targets = tracker.update(detections_in_frame_torch, img_info, img_size)

                # Update tracked sequences
                for t in online_targets:
                    track_id = t.track_id
                    if track_id not in tracked_sequences:
                        tracked_sequences[track_id] = []

                    # Find the closest pose to the tracked object
                    tlwh = t.tlwh
                    center_x, center_y = tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2
                    closest_pose = min(detections_in_frame, key=lambda p: self.distance_to_center(p[0], center_x, center_y))

                    tracked_sequences[track_id].append((frame_num, closest_pose))
            else:
                print(f"No detections for frame {frame_num}")
            pbar.update(1)

        pbar.close()

        # TODO in the sequence of all tracked poses, vote on the gender of the tracked person based on the majority of
        #  highest confidence gender detections, and assign this gender to the track




        # TODO reduce the tracks down to only two people per frame. There should only be 1 man and 1 woman tracked
        #  through the sequence. Once this is done, save the lead and follow tracks to the lead.json and follow.json



    @staticmethod
    def distance_to_center(pose, center_x, center_y):
        valid_points = [p[:2] for p in pose if p[2] > 0]
        if not valid_points:
            return float('inf')
        pose_center_x = sum(p[0] for p in valid_points) / len(valid_points)
        pose_center_y = sum(p[1] for p in valid_points) / len(valid_points)
        return ((pose_center_x - center_x) ** 2 + (pose_center_y - center_y) ** 2) ** 0.5

    #region UTILITY

    @staticmethod
    def load_json(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}

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

            poses = self.detections.get(str(frame_count), [])
            for pose in poses:
                self.draw_pose(frame, pose["keypoints"], (128, 128, 128))  # Gray for all poses

            lead_pose = self.lead.get(str(frame_count))
            if lead_pose:
                self.draw_pose(frame, lead_pose, (255, 0, 0), is_lead_or_follow=True)  # Blue for lead man

            follow_pose = self.follow.get(str(frame_count))
            if follow_pose:
                self.draw_pose(frame, follow_pose, (0, 255, 0), is_lead_or_follow=True)  # Green for follow woman

            out.write(frame)

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
