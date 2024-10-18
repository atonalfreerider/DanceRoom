import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import tqdm


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

        model = YOLO('yolo11x-man-woman.pt')

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
        frame_count = len(os.listdir(self.figure_mask_dir))
        pbar = tqdm.tqdm(total=frame_count, desc="Tracking poses")

        track_gender_votes = {}
        lead_track_ids = set()
        follow_track_ids = set()

        # First pass: Assign genders and collect votes
        for frame_num in range(frame_count):
            detections_in_frame = self.detections.get(str(frame_num), [])
            men_women_in_frame = self.men_women.get(str(frame_num), {'men': [], 'women': []})

            for detection in detections_in_frame:
                bbox = detection.get('bbox', [])
                track_id = detection.get('id')
                x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # Center of the bbox

                # Match the box to the closest gender bbox and assign the gender and confidence
                min_distance = float('inf')
                gender = 'unknown'
                confidence = 0

                for man in men_women_in_frame['men']:
                    man_center_x, man_center_y = (man[0] + man[2]) / 2, (man[1] + man[3]) / 2
                    distance = ((x - man_center_x) ** 2 + (y - man_center_y) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        gender = 'male'
                        confidence = man[4]

                for woman in men_women_in_frame['women']:
                    woman_center_x, woman_center_y = (woman[0] + woman[2]) / 2, (woman[1] + woman[3]) / 2
                    distance = ((x - woman_center_x) ** 2 + (y - woman_center_y) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        gender = 'female'
                        confidence = woman[4]

                detection['gender'] = {'gender': gender, 'confidence': confidence}

                # Collect votes for track gender
                if track_id not in track_gender_votes:
                    track_gender_votes[track_id] = {'male': 0, 'female': 0, 'unknown': 0}
                track_gender_votes[track_id][gender] += confidence

            pbar.update(1)

        pbar.close()

        # Determine final gender for each track
        for track_id, votes in track_gender_votes.items():
            if votes['male'] > votes['female'] and votes['male'] > votes['unknown']:
                lead_track_ids.add(track_id)
            elif votes['female'] > votes['male'] and votes['female'] > votes['unknown']:
                follow_track_ids.add(track_id)
            # If 'unknown' has the highest vote, we don't assign it to either lead or follow

        # Second pass: Select lead and follow poses
        lead_poses = {}
        follow_poses = {}

        for frame_num in range(frame_count):
            detections_in_frame = self.detections.get(str(frame_num), [])
            
            lead_candidates = [d for d in detections_in_frame if d.get('id') in lead_track_ids]
            follow_candidates = [d for d in detections_in_frame if d.get('id') in follow_track_ids]

            # Select the most confident lead and follow poses
            if lead_candidates:
                lead_pose = max(lead_candidates, key=lambda x: x['gender']['confidence'])
                lead_poses[frame_num] = lead_pose['keypoints']

            if follow_candidates:
                follow_pose = max(follow_candidates, key=lambda x: x['gender']['confidence'])
                follow_poses[frame_num] = follow_pose['keypoints']

        # Save the results
        self.save_json(lead_poses, self.lead_file)
        self.save_json(follow_poses, self.follow_file)

        print(f"Tracked lead and follow poses for {frame_count} frames.")

    @staticmethod
    def distance_to_center(bbox, center_x, center_y):
        pose_center_x = (bbox[0] + bbox[2]) / 2
        pose_center_y = (bbox[1] + bbox[3]) / 2
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
        lead_track = self.load_json(self.lead_file)
        follow_track = self.load_json(self.follow_file)

        # Restructure detections for efficient access
        detections_by_frame = {int(frame_id): detections for frame_id, detections in self.detections.items()}

        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw all tracked poses for the current frame
            detections_in_frame = detections_by_frame.get(frame_count, [])
            for detection in detections_in_frame:
                pose = detection.get('keypoints')
                gender = detection.get('gender')
                track_id = detection.get('id')
                bbox = detection.get('bbox')

                if pose:
                    color = (125, 125, 125)
                    if gender['gender'] == 'female':
                        color = (100, 100, 255)
                    elif gender['gender'] == 'male':
                        color = (255, 0, 0)
                    self.draw_pose(frame, pose, color)

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Put track ID text
                    if track_id is not None:
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

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames for debug video")

        cap.release()
        out.release()

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

