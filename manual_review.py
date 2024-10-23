import cv2
import json
import numpy as np
from pathlib import Path
import tkinter as tk
from collections import OrderedDict
from matplotlib import cm
import os
from pose_data_utils import PoseDataUtils
import time

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
        self.show_depth = False
        self.depth_dir = os.path.join(output_dir, 'depth')
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ui_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.draw_ui_overlay()

        self.pose_utils = PoseDataUtils()

        self.load_data()
        self.setup_gui()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.create_trackbar()

        self.save_requested = False

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
        self.lead = self.pose_utils.load_poses(self.lead_file) if self.lead_file.exists() else {}
        self.follow = self.pose_utils.load_poses(self.follow_file) if self.follow_file.exists() else {}

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Save JSON")
        self.root.geometry("200x50")
        self.save_button = tk.Button(self.root, text="Save to JSON", command=self.save_json_files)
        self.save_button.pack(pady=10)
        self.root.withdraw()  # Hide the window initially

    def save_json_files(self):
        self._save_json_files()

    def _save_json_files(self):
        try:
            # Check if lead and follow data are not empty before saving
            if self.lead:
                self.pose_utils.save_poses(self.lead, self.frame_count, self.lead_file)
                print(f"Saved lead tracks to {self.lead_file}")
            else:
                print("Lead data is empty. Skipping save for lead.")

            if self.follow:
                self.pose_utils.save_poses(self.follow, self.frame_count, self.follow_file)
                print(f"Saved follow tracks to {self.follow_file}")
            else:
                print("Follow data is empty. Skipping save for follow.")

            # Save detections-modified.json
            if self.detections:
                detections_modified_file = self.output_dir / "detections-modified.json"
                with open(detections_modified_file, 'w') as f:
                    json.dump(self.detections, f, indent=2)
                print(f"Saved modified detections to {detections_modified_file}")
            else:
                print("Detections data is empty. Skipping save for detections.")

            print("Save completed successfully")
        except Exception as e:
            print(f"Error during save: {str(e)}")

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
        if self.show_depth:
            depth_map = self.load_depth_map(self.current_frame)
            if depth_map is not None:
                frame = self.get_colored_depth_map(depth_map)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            else:
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        else:
            frame = self.get_frame(self.current_frame)
        
        if frame is None:
            return

        frame = frame.copy()  # Create a copy to avoid modifying the original

        # Draw all poses from detections
        for detection in self.detections.get(str(self.current_frame), []):
            color = self.unassigned_color
            is_lead_or_follow = False
            if self.current_frame in self.lead and self.lead[self.current_frame] and detection['id'] == self.lead[self.current_frame]['id']:
                color = self.lead_color
                is_lead_or_follow = True
            elif self.current_frame in self.follow and self.follow[self.current_frame] and detection['id'] == self.follow[self.current_frame]['id']:
                color = self.follow_color
                is_lead_or_follow = True
            self.draw_pose(frame, detection['keypoints'], color, is_lead_or_follow)

        # Highlight hovered pose
        if self.hovered_pose:
            self.draw_pose(frame, self.hovered_pose['keypoints'], self.hover_color)

        # Draw the "Save to JSON" button
        button_top = self.frame_height - self.button_height - 10
        button_left = self.frame_width - self.button_width - 10
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

        # Add the UI overlay to the frame
        frame = cv2.addWeighted(frame, 1, self.ui_overlay, 1, 0)

        # Resize the frame to fit the screen if necessary
        if frame.shape[0] != self.screen_height or frame.shape[1] != self.screen_width:
            frame = cv2.resize(frame, (self.screen_width, self.screen_height))

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
        # Adjust mouse coordinates if frame is resized
        frame_height, frame_width = self.frame_height, self.frame_width
        x = int(x * (frame_width / self.screen_width))
        y = int(y * (frame_height / self.screen_height))

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
            button_top = self.frame_height - self.button_height - 10
            button_left = self.frame_width - self.button_width - 10
            if button_left <= x <= button_left + self.button_width and button_top <= y <= button_top + self.button_height:
                self.save_json_files()
                return

            self.dragging_keypoint = self.find_closest_keypoint(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_keypoint = None
            self.draw_frame()  # Redraw the frame when we stop dragging

    def assign_role(self, role):
        if self.hovered_pose:
            current_frame_int = int(self.current_frame)  # Ensure the frame key is an integer

            if role == 'lead':
                # Assign the hovered_pose to the lead role for the current frame
                self.lead[current_frame_int] = self.hovered_pose

                # If the current frame is also in follow, remove the pose with the same id from follow
                if current_frame_int in self.follow and self.follow[current_frame_int]['id'] == self.hovered_pose['id']:
                    del self.follow[current_frame_int]  # Remove the pose from follow if it matches

            elif role == 'follow':
                # Assign the hovered_pose to the follow role for the current frame
                self.follow[current_frame_int] = self.hovered_pose

                # If the current frame is also in lead, remove the pose with the same id from lead
                if current_frame_int in self.lead and self.lead[current_frame_int]['id'] == self.hovered_pose['id']:
                    del self.lead[current_frame_int]  # Remove the pose from lead if it matches

            # Redraw the frame after the assignment
            self.draw_frame()

    def mirror_pose(self, pose):
        # Define the pairs of keypoints to be swapped
        swap_pairs = [
            (1, 2),   # Left Eye, Right Eye
            (3, 4),   # Left Ear, Right Ear
            (5, 6),   # Left Shoulder, Right Shoulder
            (7, 8),   # Left Elbow, Right Elbow
            (9, 10),  # Left Wrist, Right Wrist
            (11, 12), # Left Hip, Right Hip
            (13, 14), # Left Knee, Right Knee
            (15, 16)  # Left Ankle, Right Ankle
        ]

        for left, right in swap_pairs:
            # Swap the positions and confidence values
            pose['keypoints'][left], pose['keypoints'][right] = pose['keypoints'][right], pose['keypoints'][left]

        return pose

    def run(self):
        try:
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
                elif key == ord('r'):  # 'R' key for mirroring
                    if self.hovered_pose:
                        # Mirror the hovered pose
                        mirrored_pose = self.mirror_pose(self.hovered_pose.copy())
                        
                        # Update the pose in self.detections
                        frame_detections = self.detections.get(str(self.current_frame), [])
                        for i, detection in enumerate(frame_detections):
                            if detection['id'] == self.hovered_pose['id']:
                                frame_detections[i] = mirrored_pose
                                break
                        self.detections[str(self.current_frame)] = frame_detections
                        
                        # Update the pose in lead or follow if it's assigned
                        if self.current_frame in self.lead and self.lead[self.current_frame] and self.lead[self.current_frame]['id'] == self.hovered_pose['id']:
                            self.lead[self.current_frame] = mirrored_pose
                        elif self.current_frame in self.follow and self.follow[self.current_frame] and self.follow[self.current_frame]['id'] == self.hovered_pose['id']:
                            self.follow[self.current_frame] = mirrored_pose
                        
                        # Update the hovered pose
                        self.hovered_pose = mirrored_pose
                        
                        self.draw_frame()
                elif key == ord('d'):  # 'D' key to toggle depth map
                    self.show_depth = not self.show_depth
                elif key == ord('t'):  # 'T' key to add new T-pose
                    self.add_t_pose()
                elif key == ord('0'):  # '0' key to unassign the hovered pose
                    self.unassign_pose()
                elif key == 0x70:  # F1 key (0x70 is the scan code for F1)
                    self.save_json_files()

                if self.playing:
                    self.current_frame = min(self.current_frame + 1, self.frame_count - 1)
                    if self.current_frame == self.frame_count - 1:
                        self.playing = False

                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Add a small delay to reduce CPU usage
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Manual review interrupted. Cleaning up...")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def create_trackbar(self):
        cv2.createTrackbar('Frame', self.window_name, 0, self.frame_count - 1, self.on_trackbar)

    def on_trackbar(self, value):
        self.current_frame = value
        self.draw_frame()

    def load_depth_map(self, frame_num):
        depth_file = os.path.join(self.depth_dir, f'{frame_num:06d}.npz')
        if os.path.exists(depth_file):
            with np.load(depth_file) as data:
                keys = list(data.keys())
                if keys:
                    return data[keys[0]]
                else:
                    print(f"Warning: No data found in {depth_file}")
                    return None
        else:
            print(f"Warning: Depth file not found: {depth_file}")
            return None

    def get_colored_depth_map(self, depth_map):
        # Normalize depth map
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)

        # Reverse the normalized depth
        reversed_depth = 1 - normalized_depth

        # Apply reversed magma colormap
        colored_depth = (cm.magma(reversed_depth) * 255).astype(np.uint8)

        # Resize to match frame dimensions
        resized_depth = cv2.resize(colored_depth, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)

        return resized_depth[:, :, :3]  # Return only RGB channels

    def draw_ui_overlay(self):
        # Clear the previous overlay
        self.ui_overlay.fill(0)
        
        # Calculate button position based on frame dimensions
        button_top = self.frame_height - self.button_height - 10
        button_left = self.frame_width - self.button_width - 10
        
        # Draw the "Save to JSON" button on the UI overlay
        cv2.rectangle(self.ui_overlay, (button_left, button_top), 
                      (button_left + self.button_width, button_top + self.button_height), 
                      self.button_color, -1)
        cv2.putText(self.ui_overlay, "Save to JSON", (button_left + 10, button_top + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.button_text_color, 2)

    def unassign_pose(self):
        if self.hovered_pose:
            if self.current_frame in self.lead:
                self.lead[self.current_frame] = [d for d in self.lead[self.current_frame] if d['id'] != self.hovered_pose['id']][0] #TODO
                if not self.lead[self.current_frame]:
                    del self.lead[self.current_frame]
            if self.current_frame in self.follow:
                self.follow[self.current_frame] = [d for d in self.follow[self.current_frame] if d['id'] != self.hovered_pose['id']][0] #TODO
                if not self.follow[self.current_frame]:
                    del self.follow[self.current_frame]
            self.draw_frame()

    def create_t_pose(self):
        # Create a T-pose in the center of the frame, facing the camera
        center_x, center_y = self.frame_width // 2, self.frame_height // 2
        t_pose = {
            'id': -1,  # Use -1 as the track ID for manually added poses
            'bbox': [0, 0, 0, 0],  # Zeroed-out bounding box [x, y, width, height]
            'confidence': 0,  # Zero confidence score
            'keypoints': [
                [center_x, center_y - 100, 1],  # Nose
                [center_x - 15, center_y - 110, 1],  # Left Eye
                [center_x + 15, center_y - 110, 1],  # Right Eye
                [center_x - 25, center_y - 105, 1],  # Left Ear
                [center_x + 25, center_y - 105, 1],  # Right Ear
                [center_x - 80, center_y - 50, 1],  # Left Shoulder
                [center_x + 80, center_y - 50, 1],  # Right Shoulder
                [center_x - 150, center_y - 50, 1],  # Left Elbow
                [center_x + 150, center_y - 50, 1],  # Right Elbow
                [center_x - 220, center_y - 50, 1],  # Left Wrist
                [center_x + 220, center_y - 50, 1],  # Right Wrist
                [center_x - 30, center_y + 100, 1],  # Left Hip
                [center_x + 30, center_y + 100, 1],  # Right Hip
                [center_x - 30, center_y + 200, 1],  # Left Knee
                [center_x + 30, center_y + 200, 1],  # Right Knee
                [center_x - 30, center_y + 300, 1],  # Left Ankle
                [center_x + 30, center_y + 300, 1],  # Right Ankle
            ]
        }
        return t_pose

    def add_t_pose(self):
        new_pose = self.create_t_pose()
        
        # Add the new pose to the detections for the current frame
        frame_key = str(self.current_frame)
        if frame_key not in self.detections:
            self.detections[frame_key] = []
        self.detections[frame_key].append(new_pose)
        
        # Set the new pose as the hovered pose
        self.hovered_pose = new_pose
        
        # Redraw the frame
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
