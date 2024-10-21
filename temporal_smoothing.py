import json
import numpy as np
from pathlib import Path
import argparse

class TemporalSmoothing:
    def __init__(self, output_dir, window_size=10):
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.lead_file = self.output_dir / "lead.json"
        self.follow_file = self.output_dir / "follow.json"
        self.detections_modified_file = self.output_dir / "detections-modified.json"
        self.smoothed_lead_file = self.output_dir / "lead_smoothed.json"
        self.smoothed_follow_file = self.output_dir / "follow_smoothed.json"

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def is_valid_keypoint(self, keypoint):
        return keypoint[0] != 0 or keypoint[1] != 0

    def smooth_keypoints(self, role_data, modified_detections):
        smoothed_data = {}
        all_frames = sorted(map(int, role_data.keys()))

        for frame in all_frames:
            frame_str = str(frame)
            if frame_str not in role_data or not role_data[frame_str]:
                continue

            pose = role_data[frame_str][0]
            smoothed_pose = pose.copy()
            smoothed_keypoints = []

            for i in range(17):  # 17 keypoints in COCO format
                window_start = max(0, frame - self.window_size // 2)
                window_end = min(max(all_frames), frame + self.window_size // 2)
                window_keypoints = []
                window_weights = []

                for f in range(window_start, window_end + 1):
                    f_str = str(f)
                    if f_str in role_data and role_data[f_str]:
                        kp = role_data[f_str][0]['keypoints'][i]
                        
                        if self.is_valid_keypoint(kp):
                            window_keypoints.append(kp)
                            # Use higher weight for the current frame
                            weight = 5.0 if f == frame else 1.0
                            window_weights.append(weight)

                if window_keypoints:
                    window_keypoints = np.array(window_keypoints)
                    window_weights = np.array(window_weights)
                    weighted_avg = np.average(window_keypoints, axis=0, weights=window_weights)
                    smoothed_keypoints.append(weighted_avg.tolist())
                elif self.is_valid_keypoint(pose['keypoints'][i]):
                    # If no valid keypoints in window but current keypoint is valid, keep it
                    smoothed_keypoints.append(pose['keypoints'][i])
                else:
                    # If no valid keypoints at all, append a zero keypoint
                    smoothed_keypoints.append([0, 0, 0])

            smoothed_pose['keypoints'] = smoothed_keypoints
            smoothed_data[frame_str] = [smoothed_pose]

        return smoothed_data

    def run(self):
        lead_data = self.load_json(self.lead_file)
        follow_data = self.load_json(self.follow_file)
        modified_detections = self.load_json(self.detections_modified_file)

        smoothed_lead = self.smooth_keypoints(lead_data, modified_detections)
        smoothed_follow = self.smooth_keypoints(follow_data, modified_detections)

        self.save_json(smoothed_lead, self.smoothed_lead_file)
        self.save_json(smoothed_follow, self.smoothed_follow_file)

        print(f"Smoothed lead data saved to: {self.smoothed_lead_file}")
        print(f"Smoothed follow data saved to: {self.smoothed_follow_file}")

def main():
    parser = argparse.ArgumentParser(description="Smooth lead and follow dancer keypoints using temporal smoothing.")
    parser.add_argument("output_dir", help="Path to the output directory containing JSON files")
    parser.add_argument("--window_size", type=int, default=10, help="Size of the moving average window")
    args = parser.parse_args()

    smoother = TemporalSmoothing(args.output_dir, args.window_size)
    smoother.run()

if __name__ == "__main__":
    main()
