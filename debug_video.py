import os
import cv2
import json
import numpy as np
from tqdm import tqdm


class DebugVideo:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.lead_file = os.path.join(output_dir, 'lead.json')
        self.follow_file = os.path.join(output_dir, 'follow.json')
        self.lead = self.load_json(self.lead_file)
        self.follow = self.load_json(self.follow_file)


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

        # Use tqdm to track progress
        with tqdm(total=total_frames, desc="Generating debug video") as pbar:
            for frame_count in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw lead and follow
                lead_pose = lead_track.get(str(frame_count))
                if lead_pose:
                    lead_keypoints = lead_pose['keypoints']
                    self.draw_pose(frame, lead_keypoints, (0, 0, 255), is_lead_or_follow=True)  # Red for lead
                    cv2.putText(frame, "LEAD", (int(lead_keypoints[0][0]), int(lead_keypoints[0][1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                follow_pose = follow_track.get(str(frame_count))
                if follow_pose:
                    follow_keypoints = follow_pose['keypoints']
                    self.draw_pose(frame, follow_keypoints, (255, 192, 203), is_lead_or_follow=True)  # Pink for follow
                    cv2.putText(frame, "FOLLOW", (int(follow_keypoints[0][0]), int(follow_keypoints[0][1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 192, 203), 2)

                out.write(frame)

                # Update progress bar
                pbar.update(1)

        cap.release()
        out.release()

    @staticmethod
    def draw_pose(image, keypoints, color, is_lead_or_follow=False):
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
            if len(keypoints) > max(connection):
                start_point, start_conf = get_point_and_conf(keypoints[connection[0]])
                end_point, end_conf = get_point_and_conf(keypoints[connection[1]])

                if start_point is not None and end_point is not None:
                    avg_conf = (start_conf + end_conf) / 2
                    color_with_alpha = tuple(int(c * avg_conf) for c in color)
                    cv2.line(overlay, start_point, end_point, color_with_alpha, line_thickness)

        for point in keypoints:
            pt, conf = get_point_and_conf(point)
            if pt is not None:
                color_with_alpha = tuple(int(c * conf) for c in color)
                cv2.circle(overlay, pt, 3, color_with_alpha, -1)

        cv2.add(image, overlay, image)

    @staticmethod
    def load_json(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}