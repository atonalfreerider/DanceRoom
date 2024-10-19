import json
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from collections import OrderedDict

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
        self.min_height_threshold = 0.5 * self.frame_height
        self.current_track_id = self.find_first_track_id()
        self.window_name = "Manual Role Assignment"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        self.screen_width = 1920
        self.screen_height = 1080
        self.num_samples = 24  # Adjust this number as needed
        self.aspect_ratio = self.frame_height / int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.button_height = 40
        self.button_width = 150
        self.button_color = (200, 200, 200)  # Light gray
        self.button_text_color = (0, 0, 0)  # Black
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.setup_gui()
        self.sample_frames = []
        self.current_collage = None
        self.recursive_samples = []
        self.recursive_depth = 0
        self.max_recursive_depth = 10  # Adjust this based on your video frame rate
        self.lead_color = (255, 0, 0)  # Blue
        self.follow_color = (255, 0, 255)  # Magenta
        self.unassigned_color = (0, 255, 0)  # Green
        self.current_hover_index = None
        self.current_track_assignments = {'lead': {}, 'follow': {}}
        self.split_point = None
        self.is_hovering = False

        self.processed_track_ids = set()
        self.lead_file = self.output_dir / "lead.json"
        self.follow_file = self.output_dir / "follow.json"
        self.last_assigned_track_id = None  # Add this line
        self.load_existing_assignments()
        self.current_track_id = self.find_next_track_id_after_last_assigned()

        self.frame_cache = {}  # Add this line to create a frame cache

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

        # Sort and deduplicate before saving
        lead_tracks = self.sort_and_deduplicate_tracks(self.lead_tracks)
        follow_tracks = self.sort_and_deduplicate_tracks(self.follow_tracks)

        with open(lead_file, 'w') as f:
            json.dump(lead_tracks, f, indent=2)
        
        with open(follow_file, 'w') as f:
            json.dump(follow_tracks, f, indent=2)

        cv2.putText(self.current_collage, "Saved!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(self.window_name, self.current_collage)
        cv2.waitKey(1000)  # Display "Saved!" message for 1 second

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

    def create_collage(self, track_id, sample_frames, is_detailed=False):
        crops = []
        for frame_idx in sample_frames:
            frame = self.get_frame(frame_idx)
            if frame is None:
                continue

            person = next((d for d in self.detections[str(frame_idx)] if d['id'] == track_id), None)
            if person is None:
                continue

            bbox = person['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            crops.append((crop, frame_idx, person))

        if not crops:
            return None

        # Calculate the optimal layout
        layout = self.calculate_optimal_layout(len(crops))
        collage = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        for (crop, frame_idx, person), (x, y, w, h) in zip(crops, layout):
            crop_resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            
            # Determine the color based on the current track assignments
            if str(frame_idx) in self.current_track_assignments['lead']:
                color = self.lead_color
            elif str(frame_idx) in self.current_track_assignments['follow']:
                color = self.follow_color
            else:
                color = self.unassigned_color

            # Draw the pose on the resized crop
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
            
            collage[y:y+h, x:x+w] = crop_resized

            # Display frame number above each sample image
            cv2.putText(collage, f"Frame: {frame_idx}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw the "Save to JSON" button
        button_top = self.screen_height - self.button_height - 10
        button_left = self.screen_width - self.button_width - 10
        cv2.rectangle(collage, (button_left, button_top), 
                      (button_left + self.button_width, button_top + self.button_height), 
                      self.button_color, -1)
        cv2.putText(collage, "Save to JSON", (button_left + 10, button_top + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.button_text_color, 2)

        # Display current track ID at the top of the collage
        cv2.putText(collage, f"Track ID: {track_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return collage

    def get_recursive_samples(self, start_frame, end_frame):
        person_frames = self.find_person_frames(self.current_track_id)
        start_idx = person_frames.index(start_frame)
        end_idx = person_frames.index(end_frame)
        
        if end_idx - start_idx <= 1:
            return [start_frame, end_frame]
        
        step = (end_idx - start_idx) / 9
        return [person_frames[int(end_idx - i * step)] for i in range(10)][::-1]

    def reset_recursive_state(self):
        self.recursive_depth = 0
        self.recursive_samples = []
        self.current_samples = self.sample_frames.copy()
        self.current_hover_index = None
        self.is_hovering = False
        self.clear_frame_cache()  # Clear the frame cache when resetting recursive state

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            layout = self.calculate_optimal_layout(len(self.current_samples))
            for i, (lx, ly, lw, lh) in enumerate(layout):
                if lx <= x < lx + lw and ly <= y < ly + lh:
                    self.current_hover_index = i
                    self.is_hovering = True
                    return
            self.is_hovering = False
            self.current_hover_index = None

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is on the "Save to JSON" button
            button_top = self.screen_height - self.button_height - 10
            button_left = self.screen_width - self.button_width - 10
            if button_left <= x <= button_left + self.button_width and button_top <= y <= button_top + self.button_height:
                self.save_json_files()
                return

            if self.recursive_depth == 0:
                if self.current_hover_index is not None and self.current_hover_index < len(self.current_samples):
                    end_frame = self.current_samples[self.current_hover_index]
                    start_frame = self.current_samples[max(0, self.current_hover_index - 1)]
                    self.recursive_samples = self.get_recursive_samples(start_frame, end_frame)
                    self.recursive_depth += 1
                    self.current_samples = self.recursive_samples
                    self.show_detailed_view()
                else:
                    # Reset all roles for this track if clicked outside any sample at top level
                    self.reset_all_roles()
                    self.show_main_collage()
            else:
                if self.current_hover_index is not None and self.current_hover_index < len(self.current_samples):
                    end_frame = self.recursive_samples[self.current_hover_index]
                    start_frame = self.recursive_samples[max(0, self.current_hover_index - 1)]
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
        detailed_collage = self.create_collage(self.current_track_id, self.recursive_samples, is_detailed=True)
        if detailed_collage is not None:
            cv2.namedWindow("Detailed View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detailed View", 1920, 1080)
            cv2.imshow("Detailed View", detailed_collage)
            cv2.setMouseCallback("Detailed View", self.mouse_callback)

    def process_tracks(self):
        while self.current_track_id is not None:
            print(f"Processing track ID: {self.current_track_id}")
            self.reset_track_variables()
            
            person_frames = self.find_person_frames(self.current_track_id)
            if len(person_frames) < self.num_samples:
                sample_frames = person_frames
            else:
                step = (len(person_frames) - 1) // (self.num_samples - 1)
                sample_frames = person_frames[::step][:self.num_samples]
            
            self.sample_frames = sample_frames
            self.current_samples = self.sample_frames
            self.show_main_collage()

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    cv2.destroyAllWindows()
                    return
                elif key == 82:  # Up arrow
                    self.assign_role_with_hover('lead')
                elif key == 84:  # Down arrow
                    self.assign_role_with_hover('follow')
                elif key == 83:  # Right arrow
                    self.update_final_tracks()
                    self.processed_track_ids.add(self.current_track_id)
                    self.current_track_id = self.find_next_unprocessed_track_id()
                    cv2.destroyWindow("Detailed View")
                    self.clear_frame_cache()
                    break
                
                # Check if main window is closed
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    return

                # Check if detailed view is closed
                if cv2.getWindowProperty("Detailed View", cv2.WND_PROP_VISIBLE) < 1:
                    self.reset_recursive_state()
                    self.show_main_collage()

        self.update_final_tracks()
        cv2.destroyAllWindows()

    def reset_track_variables(self):
        self.current_track_assignments = {'lead': {}, 'follow': {}}
        self.split_point = None
        self.recursive_depth = 0
        self.recursive_samples = []
        self.current_samples = []
        self.current_hover_index = None
        self.is_hovering = False

    def show_main_collage(self):
        self.current_collage = self.create_collage(self.current_track_id, self.sample_frames)
        if self.current_collage is not None:
            cv2.imshow(self.window_name, self.current_collage)

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
                self.assign_role(role, start_frame, is_detailed=True)
            else:
                # If not hovering in detailed view, do nothing
                return

        self.update_collage()

    def update_collage(self):
        if self.recursive_depth == 0:
            self.show_main_collage()
        else:
            self.show_detailed_view()

    def assign_role(self, role, start_frame, is_detailed=False):
        other_role = 'follow' if role == 'lead' else 'lead'
        
        for frame, detections in self.detections.items():
            frame_num = int(frame)
            for detection in detections:
                if detection['id'] == self.current_track_id and self.is_valid_detection(detection):
                    if is_detailed:
                        if frame_num == start_frame:
                            # Assign the new role to the hovered frame
                            self.current_track_assignments[role][frame] = detection
                            self.current_track_assignments[other_role].pop(frame, None)
                        elif frame_num > start_frame:
                            # Assign the new role to all following frames
                            self.current_track_assignments[role][frame] = detection
                            self.current_track_assignments[other_role].pop(frame, None)
                        else:
                            # Assign the inverse role to all preceding unassigned frames
                            if frame not in self.current_track_assignments[role] and frame not in self.current_track_assignments[other_role]:
                                self.current_track_assignments[other_role][frame] = detection
                    else:
                        if frame_num >= start_frame:
                            # Assign the new role from the start_frame onwards
                            self.current_track_assignments[role][frame] = detection
                            self.current_track_assignments[other_role].pop(frame, None)
                        else:
                            # For frames before start_frame, keep existing assignments
                            if frame in self.current_track_assignments[role] or frame in self.current_track_assignments[other_role]:
                                continue
                            else:
                                # If no existing assignment, assign to the current role
                                self.current_track_assignments[role][frame] = detection

        # Ensure continuity in assignments
        self.ensure_role_continuity()

    def ensure_role_continuity(self):
        frames = sorted(list(self.detections.keys()), key=int)
        last_role = None
        last_frame = None
        
        for frame in frames:
            if frame in self.current_track_assignments['lead']:
                if last_role == 'follow' and last_frame is not None:
                    # Fill gap between last_frame and current frame with last_role
                    self.fill_gap(last_frame, int(frame), last_role)
                last_role = 'lead'
                last_frame = int(frame)
            elif frame in self.current_track_assignments['follow']:
                if last_role == 'lead' and last_frame is not None:
                    # Fill gap between last_frame and current frame with last_role
                    self.fill_gap(last_frame, int(frame), last_role)
                last_role = 'follow'
                last_frame = int(frame)
        
        # Fill any remaining frames after the last assignment
        if last_role and last_frame:
            self.fill_gap(last_frame, int(frames[-1]) + 1, last_role)

    def fill_gap(self, start_frame, end_frame, role):
        for frame in range(start_frame + 1, end_frame):
            if str(frame) not in self.current_track_assignments['lead'] and str(frame) not in self.current_track_assignments['follow']:
                for detection in self.detections[str(frame)]:
                    if detection['id'] == self.current_track_id and self.is_valid_detection(detection):
                        self.current_track_assignments[role][str(frame)] = detection
                        break

    def reset_all_roles(self):
        self.current_track_assignments = {'lead': {}, 'follow': {}}

    def update_final_tracks(self):
        for role in ['lead', 'follow']:
            for frame, detection in self.current_track_assignments[role].items():
                if frame not in self.lead_tracks:
                    self.lead_tracks[frame] = []
                if frame not in self.follow_tracks:
                    self.follow_tracks[frame] = []
                
                if role == 'lead':
                    self.lead_tracks[frame] = [detection]
                    self.follow_tracks[frame] = [d for d in self.follow_tracks[frame] if d['id'] != self.current_track_id]
                else:
                    self.follow_tracks[frame] = [detection]
                    self.lead_tracks[frame] = [d for d in self.lead_tracks[frame] if d['id'] != self.current_track_id]

        # Sort and deduplicate lead and follow tracks
        self.lead_tracks = self.sort_and_deduplicate_tracks(self.lead_tracks)
        self.follow_tracks = self.sort_and_deduplicate_tracks(self.follow_tracks)

    def load_existing_assignments(self):
        if self.lead_file.exists() and self.follow_file.exists():
            with open(self.lead_file, 'r') as f:
                self.lead_tracks = json.load(f)
            with open(self.follow_file, 'r') as f:
                self.follow_tracks = json.load(f)

            # Sort and deduplicate lead and follow tracks
            self.lead_tracks = self.sort_and_deduplicate_tracks(self.lead_tracks)
            self.follow_tracks = self.sort_and_deduplicate_tracks(self.follow_tracks)

            # Find the last assigned track ID
            last_assigned_id = -1
            for role in ['lead', 'follow']:
                for frame, detections in getattr(self, f'{role}_tracks').items():
                    for detection in detections:
                        track_id = detection['id']
                        self.processed_track_ids.add(track_id)
                        last_assigned_id = max(last_assigned_id, track_id)
            
            self.last_assigned_track_id = last_assigned_id if last_assigned_id != -1 else None

    def find_next_track_id_after_last_assigned(self):
        all_track_ids = self.get_sorted_track_ids()
        if self.last_assigned_track_id is None:
            return all_track_ids[0] if all_track_ids else None
        
        for track_id in all_track_ids:
            if track_id > self.last_assigned_track_id:
                return track_id
        
        return None  # All tracks have been processed

    def get_sorted_track_ids(self):
        all_track_ids = set()
        for detections in self.detections.values():
            for detection in detections:
                if self.is_valid_detection(detection):
                    all_track_ids.add(detection['id'])
        return sorted(list(all_track_ids))

    def find_next_unprocessed_track_id(self):
        all_track_ids = self.get_sorted_track_ids()
        for track_id in all_track_ids:
            if track_id not in self.processed_track_ids:
                return track_id
        return None  # All tracks have been processed

    def find_all_track_ids(self):
        return self.get_sorted_track_ids()

    def get_frame(self, frame_idx):
        if frame_idx not in self.frame_cache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.frame_cache[frame_idx] = frame
            else:
                return None

        return self.frame_cache[frame_idx]

    def clear_frame_cache(self):
        self.frame_cache.clear()

    def sort_and_deduplicate_tracks(self, tracks):
        # Convert frame numbers to integers for proper sorting
        sorted_tracks = OrderedDict(sorted(tracks.items(), key=lambda x: int(x[0])))
        
        # Deduplicate entries
        deduplicated_tracks = OrderedDict()
        for frame, detections in sorted_tracks.items():
            unique_detections = []
            seen_ids = set()
            for detection in detections:
                if detection['id'] not in seen_ids:
                    unique_detections.append(detection)
                    seen_ids.add(detection['id'])
            if unique_detections:
                deduplicated_tracks[frame] = unique_detections
        
        return deduplicated_tracks

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