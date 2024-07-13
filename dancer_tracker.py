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

            # Match the gender identification with each pose in the frame
            men_women_in_frame = self.men_women.get(frame_num, {'men': [], 'women': []})
            for detection in detections_in_frame:
                keypoints = detection.get('keypoints', [])
                if keypoints:
                    x, y = keypoints[0][0], keypoints[0][1]  # Using the first keypoint as reference
                    for man in men_women_in_frame['men']:
                        if man[0] <= x <= man[2] and man[1] <= y <= man[3]:
                            detection['gender'] = {'gender': 'male', 'confidence': man[4]}
                            break
                    else:
                        for woman in men_women_in_frame['women']:
                            if woman[0] <= x <= woman[2] and woman[1] <= y <= woman[3]:
                                detection['gender'] = {'gender': 'female', 'confidence': woman[4]}
                                break
                        else:
                            detection['gender'] = {'gender': 'unknown', 'confidence': 0}
                else:
                    detection['gender'] = {'gender': 'unknown', 'confidence': 0}

            # convert to torch tensor (only bbox and score)
            detections_in_frame_torch = torch.tensor(
                [[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 1.0] for d in detections_in_frame], # score is 1.0. not sure if this will affect results
                dtype=torch.float32)

            if len(detections_in_frame_torch) > 0:
                img_info = [frame_height, frame_width]
                img_size = [frame_height, frame_width]
                online_targets = tracker.update(detections_in_frame_torch, img_info, img_size)

                # Update tracked sequences
                for t in online_targets:
                    track_id = t.track_id
                    if track_id not in tracked_sequences:
                        tracked_sequences[track_id] = []

                    tlwh = t.tlwh
                    center_x, center_y = tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2
                    closest_pose = min(detections_in_frame,
                                       key=lambda p: self.distance_to_center(p['keypoints'], center_x, center_y))

                    tracked_sequences[track_id].append((frame_num, closest_pose))
            else:
                print(f"No detections for frame {frame_num}")
            pbar.update(1)

        pbar.close()

        # Vote on the gender of the tracked person based on the majority of highest confidence gender detections
        for track_id, sequence in tracked_sequences.items():
            gender_votes = {'male': 0, 'female': 0}
            for _, pose in sequence:
                gender_info = pose['gender']
                if gender_info['gender'] in gender_votes:
                    gender_votes[gender_info['gender']] += gender_info['confidence']

            tracked_sequences[track_id] = {
                'gender': max(gender_votes, key=gender_votes.get),
                'sequence': sequence
            }

        # Reduce the tracks down to only two people per frame (1 man and 1 woman)
        lead_track = None
        follow_track = None
        max_male_length = 0
        max_female_length = 0

        for track_id, track_info in tracked_sequences.items():
            if track_info['gender'] == 'male' and len(track_info['sequence']) > max_male_length:
                lead_track = track_id
                max_male_length = len(track_info['sequence'])
            elif track_info['gender'] == 'female' and len(track_info['sequence']) > max_female_length:
                follow_track = track_id
                max_female_length = len(track_info['sequence'])

        # Save the lead and follow tracks
        lead_data = {}
        follow_data = {}

        if lead_track:
            for frame_num, pose in tracked_sequences[lead_track]['sequence']:
                lead_data[str(frame_num)] = pose['keypoints']

        if follow_track:
            for frame_num, pose in tracked_sequences[follow_track]['sequence']:
                follow_data[str(frame_num)] = pose['keypoints']

        self.save_json(lead_data, self.lead_file)
        self.save_json(follow_data, self.follow_file)
        self.save_json(tracked_sequences, os.path.join(self.output_dir, 'tracked_sequences.json'))

        print(f"Saved lead and follow tracks. Lead frames: {len(lead_data)}, Follow frames: {len(follow_data)}")

    @staticmethod
    def distance_to_center(keypoints, center_x, center_y):
        valid_points = [p[:2] for p in keypoints if p[2] > 0]
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

        # Load tracked sequences
        tracked_sequences = self.load_json(os.path.join(self.output_dir, 'tracked_sequences.json'))
        lead_track = self.load_json(self.lead_file)
        follow_track = self.load_json(self.follow_file)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw all tracked poses
            for track_id, track_info in tracked_sequences.items():
                pose = next((p for f, p in track_info['sequence'] if f == frame_count), None)
                if pose:
                    color = (255, 0, 255) if track_info['gender'] == 'female' else (255, 0, 0)
                    self.draw_pose(frame, pose['keypoints'], color)

                    # Draw bounding box
                    x1, y1, x2, y2 = pose['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Put track ID text
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw lead and follow
            lead_pose = lead_track.get(str(frame_count))
            if lead_pose:
                self.draw_pose(frame, lead_pose, (0, 0, 255), is_lead_or_follow=True)  # Red for lead
                cv2.putText(frame, "LEAD", (int(lead_pose[0][0]), int(lead_pose[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            follow_pose = follow_track.get(str(frame_count))
            if follow_pose:
                self.draw_pose(frame, follow_pose, (255, 192, 203), is_lead_or_follow=True)  # Pink for follow
                cv2.putText(frame, "FOLLOW", (int(follow_pose[0][0]), int(follow_pose[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 192, 203), 2)

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

        line_thickness = 3 if is_lead_or_follow else 1

        for connection in connections:
            start_point = tuple(map(int, pose[connection[0]][:2]))
            end_point = tuple(map(int, pose[connection[1]][:2]))
            cv2.line(image, start_point, end_point, color, line_thickness)

        for point in pose:
            pt = tuple(map(int, point[:2]))
            cv2.circle(image, pt, 3, color, -1)

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
