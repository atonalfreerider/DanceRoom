import json
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from collections import OrderedDict
import gc
from pose_data_utils import PoseDataUtils


class ManualRoleAssignment:
    def __init__(self, input_video, detections_file, output_dir):
        self.input_video = input_video
        self.detections_file = detections_file
        self.output_dir = Path(output_dir)
        self.detections = self.load_json(detections_file)
        self.cap = cv2.VideoCapture(input_video)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.lead_tracks = {}
        self.follow_tracks = {}
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.min_height_threshold = 0.5 * self.frame_height
        self.current_track_id: int = -1
        self.window_name = "Manual Role Assignment"
        self.screen_width = 1920
        self.screen_height = 1080
        self.button_height = 40
        self.button_width = 150
        self.button_color = (200, 200, 200)
        self.button_text_color = (0, 0, 0)
        self.lead_color = (0, 0, 255)  # Red for lead
        self.follow_color = (255, 0, 255)  # Magenta for follow
        self.unassigned_color = (0, 255, 0)  # Green for unassigned
        self.num_samples = 24
        self.aspect_ratio = self.frame_height / self.frame_width
        self.current_hover_index = None
        self.is_hovering = False
        self.current_track_assignments = {'lead': OrderedDict(), 'follow': OrderedDict()}
        self.processed_track_ids = set()
        self.lead_file = self.output_dir / "lead.json"
        self.follow_file = self.output_dir / "follow.json"
        self.last_assigned_track_id: int = -1
        self.all_track_ids = self.find_all_track_ids()
        self.current_track_id: int = -1
        self.frame_cache = OrderedDict()
        self.max_cache_size = 100
        self.detailed_view_active = False
        self.ui_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.pose_utils = PoseDataUtils()
        self.current_collage = None
        self.collage_needs_update = True
        self.sample_frames = []
        self.current_samples = []
        self.recursive_depth = 0
        self.recursive_samples = []

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.setup_gui()
        self.load_existing_assignments()
        self.draw_ui_overlay()
        
        gc.enable()

    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r') as f:
            detections = json.load(f)

        # Convert all frame keys to integers
        return {int(frame): data for frame, data in detections.items()}

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Save JSON")
        self.root.geometry("200x50")
        self.save_button = tk.Button(self.root, text="Save to JSON", command=self.save_json_files)
        self.save_button.pack(pady=10)
        self.root.withdraw()  # Hide the window initially

    def save_json_files(self):
        try:
            # Update final tracks with current assignments before saving
            self.update_final_tracks()

            # Use PoseDataUtils to save the tracks
            self.pose_utils.save_poses(self.lead_tracks, self.frame_count, self.lead_file)
            print(f"Saved lead tracks to {self.lead_file}")

            self.pose_utils.save_poses(self.follow_tracks, self.frame_count, self.follow_file)
            print(f"Saved follow tracks to {self.follow_file}")

        except Exception as e:
            print(f"Error during save: {str(e)}")

        # Don't reload assignments or update the collage after saving
        self.collage_needs_update = True  # Just to refresh the UI to show save was successful

    def find_all_track_ids(self):
        all_track_ids = set()
        for detections in self.detections.values():
            for detection in detections:
                if self.is_valid_detection(detection):
                    all_track_ids.add(detection['id'])
        return sorted(list(all_track_ids))

    def find_next_unassigned_track_id(self):
        start_index = 0
        if self.last_assigned_track_id is not None:
            try:
                start_index = self.all_track_ids.index(self.last_assigned_track_id) + 1
            except ValueError:
                # If last_assigned_track_id is not in all_track_ids, start from the beginning
                pass

        for track_id in self.all_track_ids[start_index:]:
            if track_id not in self.processed_track_ids:
                return track_id
        return None

    def find_person_frames(self, track_id):
        person_frames = []
        for frame, detections in self.detections.items():
            for detection in detections:
                if detection['id'] == track_id and self.is_valid_detection(detection):
                    person_frames.append(int(frame))
        return sorted(person_frames)

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

    def calculate_optimal_layout(self, num_images):
        aspect_ratio = self.aspect_ratio
        screen_aspect = self.screen_width / self.screen_height
        
        # Calculate the number of rows and columns
        num_cols = int(np.ceil(np.sqrt(num_images * screen_aspect / aspect_ratio)))
        num_rows = int(np.ceil(num_images / num_cols))
        
        # Calculate image size
        img_width = self.screen_width // num_cols
        img_height = self.screen_height // num_rows
        
        # Adjust image size to maintain aspect ratio
        if img_width / img_height > aspect_ratio:
            img_width = int(img_height * aspect_ratio)
        else:
            img_height = int(img_width / aspect_ratio)
        
        # Calculate layout
        layout = []
        for i in range(num_images):
            row = i // num_cols
            col = i % num_cols
            x = col * (self.screen_width // num_cols) + (self.screen_width // num_cols - img_width) // 2
            y = row * (self.screen_height // num_rows) + (self.screen_height // num_rows - img_height) // 2
            layout.append((x, y, img_width, img_height))
        
        return layout

    def create_collage(self, track_id: int, sample_frames):
        crops = []
        for frame_idx in sample_frames:
            frame = self.get_frame(frame_idx)
            if frame is None:
                continue

            # Convert frame_idx to int and use it directly as the key
            frame_idx_int = int(frame_idx)
            if frame_idx_int not in self.detections:
                print(f"Frame {frame_idx_int} not found in detections. Skipping.")
                continue

            # Get the person from detections for the given frame
            person = next((d for d in self.detections[frame_idx_int] if d['id'] == track_id), None)
            if person is None:
                continue

            bbox = person['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2].copy()
            crops.append((crop, frame_idx, person))

        if not crops:
            return None

        layout = self.calculate_optimal_layout(len(crops))
        collage = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        for (crop, frame_idx, person), (x, y, w, h) in zip(crops, layout):
            crop_resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

            frame_idx_int = int(frame_idx)

            if frame_idx_int in self.current_track_assignments['lead'] and track_id == \
                    int(self.current_track_assignments['lead'][frame_idx_int]['id']):
                color = self.lead_color
            elif frame_idx_int in self.current_track_assignments['follow'] and track_id == \
                    int(self.current_track_assignments['follow'][frame_idx_int]['id']):
                color = self.follow_color
            else:
                color = self.unassigned_color

            keypoints = person['keypoints']
            bbox = person['bbox']
            adjusted_keypoints = [
                [kp[0] - bbox[0], kp[1] - bbox[1]] + kp[2:] for kp in keypoints
            ]
            scale_x = w / (bbox[2] - bbox[0])
            scale_y = h / (bbox[3] - bbox[1])
            scaled_keypoints = [
                [kp[0] * scale_x, kp[1] * scale_y] + kp[2:] for kp in adjusted_keypoints
            ]
            self.draw_pose(crop_resized, scaled_keypoints, color)

            collage[y:y + h, x:x + w] = crop_resized
            cv2.putText(collage, f"Frame: {frame_idx_int}", (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                        2)

        button_top = self.screen_height - self.button_height - 10
        button_left = self.screen_width - self.button_width - 10
        cv2.rectangle(collage, (button_left, button_top),
                      (button_left + self.button_width, button_top + self.button_height),
                      self.button_color, -1)
        cv2.putText(collage, "Save to JSON", (button_left + 10, button_top + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.button_text_color, 2)

        cv2.putText(collage, f"Track ID: {track_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        collage = cv2.addWeighted(collage, 1, self.ui_overlay, 1, 0)

        return collage

    def get_recursive_samples(self, start_frame, end_frame):
        person_frames = self.find_person_frames(self.current_track_id)
        start_idx = person_frames.index(start_frame)
        end_idx = person_frames.index(end_frame)

        frames_between = person_frames[start_idx:end_idx + 1]

        if len(frames_between) <= 10:
            return frames_between
        else:
            step = (len(frames_between) - 1) / 9
            return [frames_between[int(i * step)] for i in range(10)]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            layout = self.calculate_optimal_layout(len(self.current_samples))
            self.current_hover_index = None
            self.is_hovering = False
            for i, (lx, ly, lw, lh) in enumerate(layout):
                if lx <= x < lx + lw and ly <= y < ly + lh:
                    self.current_hover_index = i
                    self.is_hovering = True
                    break

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is on the "Save to JSON" button
            button_top = self.screen_height - self.button_height - 10
            button_left = self.screen_width - self.button_width - 10
            if button_left <= x <= button_left + self.button_width and button_top <= y <= button_top + self.button_height:
                self.save_json_files()
                return

            layout = self.calculate_optimal_layout(len(self.current_samples))
            for i, (lx, ly, lw, lh) in enumerate(layout):
                if lx <= x < lx + lw and ly <= y < ly + lh:
                    self.current_hover_index = i
                    self.is_hovering = True
                    break
            
            if self.is_hovering and self.current_hover_index is not None:
                hover_index = self.current_hover_index
                self.reset_detail_view()
                self.current_hover_index = hover_index
                self.is_hovering = True
                self.handle_recursive_detail()

    def show_detailed_view(self):
        detailed_collage = self.create_collage(self.current_track_id, self.current_samples)
        if detailed_collage is not None:
            cv2.namedWindow("Detailed View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detailed View", 1920, 1080)
            cv2.imshow("Detailed View", detailed_collage)
            cv2.setMouseCallback("Detailed View", self.detail_mouse_callback)
            self.detailed_view_active = True

    def process_tracks(self):
        while self.current_track_id is not None:
            self.reset_track_variables()

            person_frames = self.find_person_frames(self.current_track_id)
            
            if len(person_frames) <= self.num_samples:
                sample_frames = person_frames
            else:
                step = (len(person_frames) - 1) / (self.num_samples - 1)
                sample_frames = [person_frames[int(i * step)] for i in range(self.num_samples)]
            
            self.sample_frames = list(dict.fromkeys(sample_frames))
            self.current_samples = self.sample_frames
            self.collage_needs_update = True

            while True:
                if self.collage_needs_update:
                    self.update_main_collage()
                    self.collage_needs_update = False

                self.draw_frame()
                key = cv2.waitKey(100) & 0xFF
                
                if key != 255:
                    if key == 27:  # ESC key
                        cv2.destroyAllWindows()
                        return
                    elif key == 82:  # Up arrow
                        self.assign_role_with_hover('lead')
                        self.collage_needs_update = True
                    elif key == 84:  # Down arrow
                        self.assign_role_with_hover('follow')
                        self.collage_needs_update = True
                    elif key == 83:  # Right arrow
                        self.update_final_tracks()
                        self.processed_track_ids.add(self.current_track_id)
                        self.last_assigned_track_id = self.current_track_id
                        old_track_id = self.current_track_id
                        self.current_track_id = self.find_next_unassigned_track_id()
                        cv2.destroyWindow("Detailed View")
                        self.reset_detail_view()
                        self.clear_frame_cache()
                        gc.collect()
                        self.collage_needs_update = True
                        break
                    elif key == ord('s'):  # 's' key for saving
                        self.save_json_files()
                
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    return

                if self.detailed_view_active:
                    if cv2.getWindowProperty("Detailed View", cv2.WND_PROP_VISIBLE) < 1:
                        self.detailed_view_active = False
                        self.reset_detail_view()
                        self.update_main_collage()
                    else:
                        detailed_key = cv2.waitKey(100) & 0xFF  # Increased wait time to 100ms
                        if detailed_key != 255:  # If a key was pressed in detailed view
                            if detailed_key == 27 or detailed_key == ord('q'):  # ESC or 'q' key to quit detailed view
                                cv2.destroyWindow("Detailed View")
                                self.reset_detail_view()
                                self.update_main_collage()
                            elif detailed_key == 82:  # Up arrow
                                self.assign_role_with_hover('lead')
                                self.update_detailed_view()
                                self.update_main_collage()
                            elif detailed_key == 84:  # Down arrow
                                self.assign_role_with_hover('follow')
                                self.update_detailed_view()
                                self.update_main_collage()

    def reset_track_variables(self):
        self.split_point = None
        self.recursive_depth = 0
        self.recursive_samples = []
        self.current_samples = []
        self.current_hover_index = None
        self.is_hovering = False

    def update_main_collage(self):
        self.current_collage = self.create_collage(self.current_track_id, self.sample_frames)
        if self.current_collage is not None:
            cv2.imshow(self.window_name, self.current_collage)
            cv2.waitKey(1)
        else:
            print(f"Failed to create collage for track ID: {self.current_track_id}")

    def assign_role_with_hover(self, role):
        if self.recursive_depth == 0:
            # On the main collage
            if self.is_hovering and self.current_hover_index is not None and self.current_hover_index < len(self.current_samples):
                # If hovering over a sample, assign from that sample onwards
                start_frame = self.sample_frames[self.current_hover_index]
                self.assign_role(role, start_frame)
            else:
                # If not hovering, assign to the entire track
                self.assign_role(role, self.sample_frames[0])
        else:
            # In detailed view
            if self.is_hovering and self.current_hover_index is not None and self.current_hover_index < len(self.current_samples):
                start_frame = self.current_samples[self.current_hover_index]
                self.assign_role(role, start_frame)
            else:
                # If not hovering in detailed view, do nothing
                return

        print(f"Role {role} assigned, updating collage...")
        self.update_collage()
        self.update_main_collage()
        self.collage_needs_update = True

    def update_collage(self):
        if self.recursive_depth == 0:
            self.update_main_collage()
        else:
            self.update_detailed_view()
        self.update_main_collage()  # Always update the main collage

    def assign_role(self, role, start_frame):
        other_role = 'follow' if role == 'lead' else 'lead'

        # Ensure frame keys are sorted and treated as integers
        frames = sorted(list(self.detections.keys()), key=int)
        start_index = frames.index(start_frame)

        for i, frame in enumerate(frames):
            frame_int = int(frame)

            for detection in self.detections[frame_int]:
                if detection['id'] == self.current_track_id and self.is_valid_detection(detection):
                    if i >= start_index:
                        # Assign the detection to the current role
                        self.current_track_assignments[role][frame_int] = detection
                        if self.current_track_assignments[other_role][frame_int]['id'] == detection['id']:
                            # unassign
                            self.current_track_assignments[other_role][frame_int] = PoseDataUtils.create_empty_pose()
                    else:
                        if self.current_track_assignments[other_role][frame_int]['id'] == -1 and self.is_role_defined_at(role, frame_int, detection['id']):
                            # gap fill
                            self.current_track_assignments[other_role][frame_int] = detection

    def is_role_defined_at(self, role, frame_int: int, track_to_check:int) -> bool:
        # Check if the role is defined for the given frame
        if frame_int in self.current_track_assignments[role]:
            detection = self.current_track_assignments[role][frame_int]
            # Check if the detection has a valid id (not an empty detection)
            if detection['id'] != -1 and detection['id'] != track_to_check:  # Assuming -1 indicates an empty or unassigned detection
                return True
        return False

    def update_final_tracks(self):
        for role in ['lead', 'follow']:
            for frame, detection in self.current_track_assignments[role].items():
                if role == 'lead':
                    self.lead_tracks[frame] = detection
                else:
                    self.follow_tracks[frame] = detection

        # Update processed_track_ids
        self.processed_track_ids.add(self.current_track_id)

    def load_existing_assignments(self):
        self.lead_file = self.output_dir / "lead.json"
        self.follow_file = self.output_dir / "follow.json"
        
        if self.lead_file.exists():
            self.lead_tracks = self.pose_utils.load_poses(self.lead_file)
        else:
            self.lead_tracks = OrderedDict()
            for frame in range(self.frame_count):
                self.lead_tracks[frame] = self.pose_utils.create_empty_pose()
        
        if self.follow_file.exists():
            self.follow_tracks = self.pose_utils.load_poses(self.follow_file)
        else:
            self.follow_tracks = OrderedDict()
            for frame in range(self.frame_count):
                self.follow_tracks[frame] = self.pose_utils.create_empty_pose()

        # Reset processed_track_ids and current_track_assignments
        self.processed_track_ids = set()
        self.current_track_assignments = {'lead': OrderedDict(), 'follow': OrderedDict()}
        
        for role, tracks in [('lead', self.lead_tracks), ('follow', self.follow_tracks)]:
            for frame, frame_detection in tracks.items():
                if frame not in self.current_track_assignments[role]:
                    self.current_track_assignments[role][frame] = PoseDataUtils.create_empty_pose()
                track_id = int(frame_detection['id'])
                if track_id == -1:
                    continue
                self.processed_track_ids.add(track_id)
                self.current_track_assignments[role][frame] = frame_detection

        # Set the last_assigned_track_id to the highest track ID that was assigned
        self.last_assigned_track_id = max(self.processed_track_ids) if self.processed_track_ids else None

        print(f"Loaded {len(self.processed_track_ids)} fully processed track IDs")
        print(f"Last assigned track ID: {self.last_assigned_track_id}")

        # Set the current_track_id to the next unassigned track
        self.current_track_id = self.find_next_unassigned_track_id()
        print(f"Starting with track ID: {self.current_track_id}")

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

    def clear_frame_cache(self):
        self.frame_cache.clear()
        gc.collect()  # Force garbage collection

    def handle_recursive_detail(self):
        if not self.is_hovering or self.current_hover_index is None or self.current_hover_index >= len(
                self.current_samples):
            return

        end_frame = self.current_samples[self.current_hover_index]
        start_frame = self.current_samples[max(0, self.current_hover_index - 1)]
        new_samples = self.get_recursive_samples(start_frame, end_frame)

        if len(new_samples) <= 1:
            return

        self.recursive_samples = new_samples
        self.current_samples = self.recursive_samples
        self.recursive_depth += 1
        self.show_detailed_view()

    def reset_detailed_view_state(self):
        self.recursive_samples = []
        self.recursive_depth = 0
        self.current_samples = self.sample_frames.copy()
        self.current_hover_index = None
        self.is_hovering = False

    def detail_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            layout = self.calculate_optimal_layout(len(self.current_samples))
            old_hover_index = self.current_hover_index
            self.current_hover_index = None
            self.is_hovering = False
            for i, (lx, ly, lw, lh) in enumerate(layout):
                if lx <= x < lx + lw and ly <= y < ly + lh:
                    self.current_hover_index = i
                    self.is_hovering = True
                    break
            
            if self.current_hover_index != old_hover_index:
                self.update_detailed_view()

        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.is_hovering and self.current_hover_index is not None and self.current_hover_index < len(self.current_samples):
                self.handle_recursive_detail()

    def update_detailed_view(self):
        detailed_collage = self.create_collage(self.current_track_id, self.current_samples)
        if detailed_collage is not None:
            # Highlight the hovered sample
            if self.is_hovering and self.current_hover_index is not None:
                layout = self.calculate_optimal_layout(len(self.current_samples))
                x, y, w, h = layout[self.current_hover_index]
                cv2.rectangle(detailed_collage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow("Detailed View", detailed_collage)

    def reset_detail_view(self):
        self.recursive_samples = []
        self.recursive_depth = 0
        if not self.detailed_view_active:
            self.current_samples = self.sample_frames.copy()
        self.detailed_view_active = False

    def draw_frame(self):
        if self.current_collage is not None:
            cv2.imshow(self.window_name, self.current_collage)

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

def main(input_video, detections_file, output_dir):
    assigner = ManualRoleAssignment(input_video, detections_file, output_dir)
    if assigner.current_track_id is None:
        print("All tracks have been processed. No manual assignment needed.")
    else:
        assigner.process_tracks()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manually assign roles to tracked persons.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("detections_file", help="Path to the detections JSON file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.detections_file, args.output_dir)
