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
        self.current_track_id = self.find_first_track_id()
        self.window_name = "Manual Role Assignment"
        cv2.namedWindow(self.window_name)
        self.setup_gui()

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
            json.dump(self.numpy_to_python(self.lead_tracks), f, indent=2)
        
        with open(follow_file, 'w') as f:
            json.dump(self.numpy_to_python(self.follow_tracks), f, indent=2)

        messagebox.showinfo("Save Complete", "Lead and Follow JSON files have been saved.")

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

    def find_first_track_id(self):
        all_track_ids = set()
        for detections in self.detections.values():
            for detection in detections:
                all_track_ids.add(detection['id'])
        return min(all_track_ids) if all_track_ids else None

    def find_next_track_id(self, current_id):
        all_track_ids = set()
        for detections in self.detections.values():
            for detection in detections:
                if detection['id'] > current_id:
                    all_track_ids.add(detection['id'])
        return min(all_track_ids) if all_track_ids else None

    def find_first_appearance(self, track_id):
        for frame, detections in self.detections.items():
            for detection in detections:
                if detection['id'] == track_id:
                    return int(frame)
        return None

    def get_person_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return frame[y1:y2, x1:x2]

    def process_tracks(self):
        lead_file = self.output_dir / "lead.json"
        follow_file = self.output_dir / "follow.json"
        
        if lead_file.exists() and follow_file.exists():
            print("Lead and Follow JSON files already exist. Skipping manual assignment.")
            return

        while self.current_track_id is not None:
            frame_index = self.find_first_appearance(self.current_track_id)
            if frame_index is None:
                self.current_track_id = self.find_next_track_id(self.current_track_id)
                continue

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if not ret:
                break

            person = next((d for d in self.detections[str(frame_index)] if d['id'] == self.current_track_id), None)
            if person is None:
                self.current_track_id = self.find_next_track_id(self.current_track_id)
                continue

            crop = self.get_person_crop(frame, person['bbox'])
            cv2.imshow(self.window_name, crop)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC key
                    cv2.destroyAllWindows()
                    self.root.destroy()
                    return
                elif key == 82:  # Up arrow
                    self.assign_role('lead')
                    break
                elif key == 84:  # Down arrow
                    self.assign_role('follow')
                    break
                elif key == 83:  # Right arrow
                    break

            self.current_track_id = self.find_next_track_id(self.current_track_id)

        cv2.destroyAllWindows()
        self.root.deiconify()  # Show the save button window
        self.root.mainloop()

    def assign_role(self, role):
        tracks = self.lead_tracks if role == 'lead' else self.follow_tracks
        for frame, detections in self.detections.items():
            for detection in detections:
                if detection['id'] == self.current_track_id:
                    if frame not in tracks:
                        tracks[frame] = []
                    tracks[frame].append(detection)

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
