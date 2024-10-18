import json
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

class ManualRoleAssignment:
    def __init__(self, input_video, detections_file, output_dir):
        self.input_video = input_video
        self.detections_file = detections_file
        self.output_dir = Path(output_dir)
        self.detections = self.load_json(detections_file)
        self.cap = cv2.VideoCapture(input_video)
        self.lead_tracks = {}
        self.follow_tracks = {}
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.min_height_threshold = 0.2 * self.frame_height
        self.current_track_id = self.find_first_track_id()
        self.window_name = "Manual Role Assignment"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.setup_gui()
        self.sample_frames = []
        self.current_collage = None
        self.crop_size = (160, 320)  # Increased size for better visibility
        self.recursive_samples = []
        self.recursive_depth = 0
        self.max_recursive_depth = 10  # Adjust this based on your video frame rate
        self.lead_color = (255, 0, 0)  # Blue
        self.follow_color = (255, 0, 255)  # Magenta
        self.unassigned_color = (0, 255, 0)  # Green
        self.current_hover_index = None

    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Save JSON")
        self.root.geometry("200x50")
        self.save_button = tk.Button(self.root, text="Save to JSON", command=self.save_json_files)
        self.save_button.pack(pady=10)
        self.root.withdraw()  # Hide the window initially

    def save_json_files(self):
        lead_file = self.output_dir / "lead.json"
        follow_file = self.output_dir / "follow.json"

        with open(lead_file, 'w') as f:
            json.dump(self.lead_tracks, f, indent=2)
        
        with open(follow_file, 'w') as f:
            json.dump(self.follow_tracks, f, indent=2)

        messagebox.showinfo("Save Complete", "Lead and Follow JSON files have been saved.")

    def find_first_track_id(self):
        all_track_ids = set()
        for detections in self.detections.values():
            for detection in detections:
                if self.is_valid_detection(detection):
                    all_track_ids.add(detection['id'])
        return min(all_track_ids) if all_track_ids else None

    def find_next_track_id(self, current_id):
        all_track_ids = set()
        for detections in self.detections.values():
            for detection in detections:
                if detection['id'] > current_id and self.is_valid_detection(detection):
                    all_track_ids.add(detection['id'])
        return min(all_track_ids) if all_track_ids else None

    def find_person_frames(self, track_id):
        person_frames = []
        for frame, detections in self.detections.items():
            for detection in detections:
                if detection['id'] == track_id and self.is_valid_detection(detection):
                    person_frames.append(int(frame))
        return sorted(person_frames)

    def get_person_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return frame[y1:y2, x1:x2]

    def is_valid_detection(self, detection):
        bbox = detection['bbox']
        height = bbox[3] - bbox[1]
        return height >= self.min_height_threshold

    def draw_pose(self, image, keypoints, color):
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]

        def get_point(kp):
            if len(kp) >= 2 and (kp[0] > 0 and kp[1] > 0):  # Changed condition here
                return (int(kp[0]), int(kp[1]))
            return None

        # Draw connections
        for connection in connections:
            start_point = get_point(keypoints[connection[0]])
            end_point = get_point(keypoints[connection[1]])
            if start_point and end_point:
                cv2.line(image, start_point, end_point, color, 3)

        # Draw keypoints
        for point in keypoints:
            pt = get_point(point)
            if pt:
                cv2.circle(image, pt, 5, color, -1)

    def create_collage(self, track_id, sample_frames):
        crops = []
        for frame_idx in sample_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                continue

            person = next((d for d in self.detections[str(frame_idx)] if d['id'] == track_id), None)
            if person is None:
                continue

            bbox = person['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, self.crop_size)
            
            # Determine the color based on the role assignment
            if str(frame_idx) in self.lead_tracks:
                color = self.lead_color
            elif str(frame_idx) in self.follow_tracks:
                color = self.follow_color
            else:
                color = self.unassigned_color

            # Draw the pose on the resized crop
            keypoints = person['keypoints']
            adjusted_keypoints = [
                [kp[0] - x1, kp[1] - y1] + kp[2:] for kp in keypoints
            ]
            scale_x = self.crop_size[0] / (x2 - x1)
            scale_y = self.crop_size[1] / (y2 - y1)
            scaled_keypoints = [
                [kp[0] * scale_x, kp[1] * scale_y] + kp[2:] for kp in adjusted_keypoints
            ]
            self.draw_pose(crop_resized, scaled_keypoints, color)
            
            crops.append(crop_resized)

        if not crops:
            return None

        # Create the collage
        rows = (len(crops) + 4) // 5  # 5 images per row, rounded up
        cols = min(5, len(crops))
        collage = np.zeros((rows * self.crop_size[1], cols * self.crop_size[0], 3), dtype=np.uint8)

        for i, crop in enumerate(crops):
            row = i // 5
            col = i % 5
            collage[row*self.crop_size[1]:(row+1)*self.crop_size[1], col*self.crop_size[0]:(col+1)*self.crop_size[0]] = crop

        return collage

    def get_recursive_samples(self, start_frame, end_frame):
        person_frames = self.find_person_frames(self.current_track_id)
        start_idx = person_frames.index(start_frame)
        end_idx = person_frames.index(end_frame)
        
        if end_idx - start_idx <= 1:
            return [start_frame, end_frame]
        
        step = (end_idx - start_idx) / 9
        return [person_frames[int(start_idx + i * step)] for i in range(10)]

    def mouse_callback(self, event, x, y, flags, param):
        row = y // self.crop_size[1]
        col = x // self.crop_size[0]
        index = row * 5 + col

        if event == cv2.EVENT_MOUSEMOVE:
            self.current_hover_index = index if index < len(self.current_samples) else None

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.recursive_depth == 0:
                if index < len(self.sample_frames):
                    if index > 0:
                        start_frame = self.sample_frames[index - 1]
                        end_frame = self.sample_frames[index]
                        self.recursive_samples = self.get_recursive_samples(start_frame, end_frame)
                        self.recursive_depth += 1
                        self.current_samples = self.recursive_samples
                        self.show_detailed_view()
            else:
                if index < len(self.recursive_samples):
                    if index > 0:
                        start_frame = self.recursive_samples[index - 1]
                        end_frame = self.recursive_samples[index]
                        new_samples = self.get_recursive_samples(start_frame, end_frame)
                        
                        if len(new_samples) == 2 or self.recursive_depth >= self.max_recursive_depth:
                            # We've reached the finest resolution or max depth
                            self.recursive_samples = new_samples
                            self.current_samples = self.recursive_samples
                            self.show_detailed_view()
                        else:
                            self.recursive_samples = new_samples
                            self.current_samples = self.recursive_samples
                            self.recursive_depth += 1
                            self.show_detailed_view()

    def show_detailed_view(self):
        detailed_collage = self.create_collage(self.current_track_id, self.recursive_samples)
        if detailed_collage is not None:
            cv2.imshow("Detailed View", detailed_collage)
            cv2.setMouseCallback("Detailed View", self.mouse_callback)

    def process_tracks(self):
        lead_file = self.output_dir / "lead.json"
        follow_file = self.output_dir / "follow.json"
        
        if lead_file.exists() and follow_file.exists():
            print("Lead and Follow JSON files already exist. Skipping manual assignment.")
            return

        while self.current_track_id is not None:
            person_frames = self.find_person_frames(self.current_track_id)
            if len(person_frames) < 10:
                sample_frames = person_frames
            else:
                step = (len(person_frames) - 1) // 9
                sample_frames = person_frames[::step][:10]
            
            self.sample_frames = sample_frames
            self.current_samples = self.sample_frames
            self.show_main_collage()

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    cv2.destroyAllWindows()
                    self.root.destroy()
                    return
                elif key == 82:  # Up arrow
                    self.assign_role_with_hover('lead')
                elif key == 84:  # Down arrow
                    self.assign_role_with_hover('follow')
                elif key == 83:  # Right arrow
                    self.current_track_id = self.find_next_track_id(self.current_track_id)
                    self.recursive_depth = 0
                    self.recursive_samples = []
                    cv2.destroyWindow("Detailed View")
                    break

        cv2.destroyAllWindows()
        self.root.deiconify()  # Show the save button window
        self.root.mainloop()

    def show_main_collage(self):
        self.current_collage = self.create_collage(self.current_track_id, self.sample_frames)
        if self.current_collage is not None:
            cv2.imshow(self.window_name, self.current_collage)

    def assign_role_with_hover(self, role):
        if self.current_hover_index is not None and self.current_hover_index < len(self.current_samples):
            start_frame = self.current_samples[self.current_hover_index]
        else:
            start_frame = self.current_samples[0]
        
        self.assign_role(role, start_frame)
        self.show_main_collage()
        if self.recursive_depth > 0:
            self.show_detailed_view()

    def assign_role(self, role, start_frame):
        tracks = self.lead_tracks if role == 'lead' else self.follow_tracks
        other_tracks = self.follow_tracks if role == 'lead' else self.lead_tracks
        for frame, detections in self.detections.items():
            frame_num = int(frame)
            if frame_num >= start_frame:
                for detection in detections:
                    if detection['id'] == self.current_track_id and self.is_valid_detection(detection):
                        if frame not in tracks:
                            tracks[frame] = []
                        tracks[frame].append(detection)
                        if frame in other_tracks:
                            other_tracks[frame] = [d for d in other_tracks[frame] if d['id'] != self.current_track_id]
            elif frame_num < start_frame:
                for detection in detections:
                    if detection['id'] == self.current_track_id and self.is_valid_detection(detection):
                        if frame not in other_tracks:
                            other_tracks[frame] = []
                        other_tracks[frame].append(detection)
                        if frame in tracks:
                            tracks[frame] = [d for d in tracks[frame] if d['id'] != self.current_track_id]

def main(input_video, detections_file, output_dir):
    assigner = ManualRoleAssignment(input_video, detections_file, output_dir)
    assigner.process_tracks()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manually assign roles to tracked persons.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("detections_file", help="Path to the detections JSON file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.detections_file, args.output_dir)
