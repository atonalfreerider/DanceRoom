import cv2
import json
import numpy as np
from pathlib import Path
import tkinter as tk
from collections import OrderedDict

class ManualReview:
    def __init__(self, input_video, detections_file, output_dir):
        self.input_video = input_video
        self.detections_file = detections_file
        self.output_dir = Path(output_dir)
        self.cap = cv2.VideoCapture(input_video)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.playing = False
        self.window_name = "Manual Review"
        self.screen_width = 1920
        self.screen_height = 1080
        self.button_height = 40
        self.button_width = 150
        self.button_color = (200, 200, 200)
        self.button_text_color = (0, 0, 0)
        self.lead_color = (0, 0, 255)  # Red for lead
        self.follow_color = (255, 0, 255)  # Magenta for follow
        self.unassigned_color = (128, 128, 128)  # Grey for unassigned
        self.hover_color = (255, 255, 0)  # Yellow for hover
        self.click_radius = 10
        self.hovered_pose = None
        self.dragging_keypoint = None
        self.frame_cache = OrderedDict()
        self.max_cache_size = 100

        self.load_data()
        self.setup_gui()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.create_trackbar()

    def load_data(self):
        # Load detections
        detections_modified_file = self.output_dir / "detections-modified.json"
        if detections_modified_file.exists():
            with open(detections_modified_file, 'r') as f:
                self.detections = json.load(f)
        else:
            # If detections-modified.json doesn't exist, create it from detections.json
            with open(self.detections_file, 'r') as f:
                self.detections = json.load(f)
            # Save the duplicate as detections-modified.json
            with open(detections_modified_file, 'w') as f:
                json.dump(self.detections, f, indent=2)
            print(f"Created {detections_modified_file} as a duplicate of {self.detections_file}")

        # Load lead and follow
        self.lead_file = self.output_dir / "lead.json"
        self.follow_file = self.output_dir / "follow.json"
        self.lead = self.load_json(self.lead_file)
        self.follow = self.load_json(self.follow_file)

    def load_json(self, file_path):
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Save JSON")
        self.root.geometry("200x50")
        self.save_button = tk.Button(self.root, text="Save to JSON", command=self.save_json_files)
        self.save_button.pack(pady=10)
        self.root.withdraw()  # Hide the window initially

    def save_json_files(self):
        # Save lead.json and follow.json
        with open(self.lead_file, 'w') as f:
            json.dump(self.lead, f, indent=2)
        with open(self.follow_file, 'w') as f:
            json.dump(self.follow, f, indent=2)

        # Save detections-modified.json
        detections_modified_file = self.output_dir / "detections-modified.json"
        with open(detections_modified_file, 'w') as f:
            json.dump(self.detections, f, indent=2)

        print(f"Saved lead tracks to {self.lead_file}")
        print(f"Saved follow tracks to {self.follow_file}")
        print(f"Saved modified detections to {detections_modified_file}")

    def get_frame(self, frame_idx):
        if frame_idx not in self.frame_cache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                if len(self.frame_cache) >= self.max_cache_size:
                    self.frame_cache.popitem(last=False)
                self.frame_cache[frame_idx] = frame
            else:
                return None
        return self.frame_cache[frame_idx]

    def draw_pose(self, image, keypoints, color, is_lead_or_follow=False):
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]

        def is_valid_point(point):
            return point[0] != 0 or point[1] != 0

        for connection in connections:
            start_point = keypoints[connection[0]][:2]
            end_point = keypoints[connection[1]][:2]
            if is_valid_point(start_point) and is_valid_point(end_point):
                cv2.line(image, tuple(map(int, start_point)), tuple(map(int, end_point)), color, 2)

        for point in keypoints:
            if is_valid_point(point[:2]):
                cv2.circle(image, tuple(map(int, point[:2])), 3, color, -1)

        if is_lead_or_follow:
            # Draw 'L' on left side
            left_shoulder = keypoints[5][:2]
            left_hip = keypoints[11][:2]
            if is_valid_point(left_shoulder) and is_valid_point(left_hip):
                mid_point = ((left_shoulder[0] + left_hip[0]) // 2, (left_shoulder[1] + left_hip[1]) // 2)
                cv2.putText(image, 'L', tuple(map(int, mid_point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw 'R' on right side
            right_shoulder = keypoints[6][:2]
            right_hip = keypoints[12][:2]
            if is_valid_point(right_shoulder) and is_valid_point(right_hip):
                mid_point = ((right_shoulder[0] + right_hip[0]) // 2, (right_shoulder[1] + right_hip[1]) // 2)
                cv2.putText(image, 'R', tuple(map(int, mid_point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_frame(self):
        frame = self.get_frame(self.current_frame).copy()  # Create a copy of the original frame
        if frame is None:
            return

        # Draw all poses from detections
        for detection in self.detections.get(str(self.current_frame), []):
            color = self.unassigned_color
            is_lead_or_follow = False
            if str(self.current_frame) in self.lead and self.lead[str(self.current_frame)] and detection['id'] == self.lead[str(self.current_frame)][0]['id']:
                color = self.lead_color
                is_lead_or_follow = True
            elif str(self.current_frame) in self.follow and self.follow[str(self.current_frame)] and detection['id'] == self.follow[str(self.current_frame)][0]['id']:
                color = self.follow_color
                is_lead_or_follow = True
            self.draw_pose(frame, detection['keypoints'], color, is_lead_or_follow)

        # Highlight hovered pose
        if self.hovered_pose:
            self.draw_pose(frame, self.hovered_pose['keypoints'], self.hover_color)

        # Draw the "Save to JSON" button
        button_top = self.screen_height - self.button_height - 10
        button_left = self.screen_width - self.button_width - 10
        cv2.rectangle(frame, (button_left, button_top), 
                      (button_left + self.button_width, button_top + self.button_height), 
                      self.button_color, -1)
        cv2.putText(frame, "Save to JSON", (button_left + 10, button_top + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.button_text_color, 2)

        # Display current frame number
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Update trackbar position
        cv2.setTrackbarPos('Frame', self.window_name, self.current_frame)

        cv2.imshow(self.window_name, frame)

    def find_closest_keypoint(self, x, y):
        closest_distance = float('inf')
        closest_pose = None
        closest_keypoint_index = None

        for detection in self.detections.get(str(self.current_frame), []):
            for i, keypoint in enumerate(detection['keypoints']):
                distance = np.sqrt((x - keypoint[0])**2 + (y - keypoint[1])**2)
                if distance < closest_distance and distance < self.click_radius:
                    closest_distance = distance
                    closest_pose = detection
                    closest_keypoint_index = i

        return (closest_pose, closest_keypoint_index)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            old_hovered_pose = self.hovered_pose
            self.hovered_pose, _ = self.find_closest_keypoint(x, y)
            
            if self.dragging_keypoint:
                pose, keypoint_index = self.dragging_keypoint
                if pose is not None and keypoint_index is not None:
                    pose['keypoints'][keypoint_index][0] = x
                    pose['keypoints'][keypoint_index][1] = y
                    
                    # Update the detection in self.detections
                    frame_detections = self.detections.get(str(self.current_frame), [])
                    for i, detection in enumerate(frame_detections):
                        if detection['id'] == pose['id']:
                            frame_detections[i] = pose
                            break
                    self.detections[str(self.current_frame)] = frame_detections
                
            # Redraw the frame if the hovered pose changed or if we're dragging
            if self.hovered_pose != old_hovered_pose or self.dragging_keypoint:
                self.draw_frame()

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is on the "Save to JSON" button
            button_top = self.screen_height - self.button_height - 10
            button_left = self.screen_width - self.button_width - 10
            if button_left <= x <= button_left + self.button_width and button_top <= y <= button_top + self.button_height:
                self.save_json_files()
                return

            self.dragging_keypoint = self.find_closest_keypoint(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_keypoint = None
            self.draw_frame()  # Redraw the frame when we stop dragging

    def assign_role(self, role):
        if self.hovered_pose:
            if role == 'lead':
                self.lead[str(self.current_frame)] = [self.hovered_pose]
                if str(self.current_frame) in self.follow:
                    self.follow[str(self.current_frame)] = [d for d in self.follow[str(self.current_frame)] if d['id'] != self.hovered_pose['id']]
            elif role == 'follow':
                self.follow[str(self.current_frame)] = [self.hovered_pose]
                if str(self.current_frame) in self.lead:
                    self.lead[str(self.current_frame)] = [d for d in self.lead[str(self.current_frame)] if d['id'] != self.hovered_pose['id']]
            self.draw_frame()

    def run(self):
        while True:
            self.draw_frame()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == 32:  # Spacebar
                self.playing = not self.playing
            elif key == 83:  # Right arrow
                self.current_frame = min(self.current_frame + 1, self.frame_count - 1)
            elif key == 81:  # Left arrow
                self.current_frame = max(self.current_frame - 1, 0)
            elif key == ord('1'):
                self.assign_role('lead')
            elif key == ord('2'):
                self.assign_role('follow')
            elif key == 13:  # Enter key
                # Get the current value from the trackbar
                new_frame = cv2.getTrackbarPos('Frame', self.window_name)
                self.current_frame = max(0, min(new_frame, self.frame_count - 1))

            if self.playing:
                self.current_frame = min(self.current_frame + 1, self.frame_count - 1)
                if self.current_frame == self.frame_count - 1:
                    self.playing = False

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def create_trackbar(self):
        cv2.createTrackbar('Frame', self.window_name, 0, self.frame_count - 1, self.on_trackbar)

    def on_trackbar(self, value):
        self.current_frame = value
        self.draw_frame()

def main(input_video, detections_file, output_dir):
    reviewer = ManualReview(input_video, detections_file, output_dir)
    reviewer.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manually review and edit pose detections.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("detections_file", help="Path to the detections JSON file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.detections_file, args.output_dir)
