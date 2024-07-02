import cv2
import numpy as np
from ultralytics import YOLO, SAM
import json
import os

class PersonSegmentation:
    def __init__(self, output_dir):
        self.seg_model = SAM('sam_b.pt')
        self.pose_model = YOLO('yolov8x-pose-p6.pt')
        self.output_dir = output_dir
        self.mask_dir = os.path.join(output_dir, 'masks')
        os.makedirs(self.mask_dir, exist_ok=True)
        self.poses = []
        self.pose_file = os.path.join(output_dir, 'poses.json')
        self.bg_video_path = os.path.join(output_dir, 'background_only.mp4')

    def load_existing_data(self):
        if os.path.exists(self.pose_file):
            with open(self.pose_file, 'r') as f:
                self.poses = json.load(f)
            return True
        return False

    def process_frame(self, frame, frame_num):
        mask_file = os.path.join(self.mask_dir, f'mask_{frame_num:06d}.png')

        # Use cached poses if available, otherwise perform pose detection
        if frame_num < len(self.poses):
            frame_poses = self.poses[frame_num]["poses"]
        else:
            # Pose detection
            pose_results = self.pose_model(frame, classes=[0])
            frame_poses = []
            for r in pose_results:
                if r.keypoints is not None:
                    for person_keypoints in r.keypoints.data:
                        keypoints = person_keypoints.cpu().numpy().tolist()
                        frame_poses.append(keypoints)
            self.poses.append({"frame": frame_num, "poses": frame_poses})

        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        else:
            # Segmentation
            seg_results = self.seg_model(frame)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for seg in seg_results:
                for instance_mask in seg.masks.data:
                    instance_mask_np = instance_mask.cpu().numpy().astype(np.uint8)

                    # Check if the segment contains any pose keypoints
                    contains_keypoints = False
                    for pose in frame_poses:
                        for kp in pose:
                            x, y, conf = kp
                            if conf > 0.01 and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                                if instance_mask_np[int(y), int(x)] > 0:
                                    contains_keypoints = True
                                    break
                        if contains_keypoints:
                            break

                    if contains_keypoints:
                        # If the segment contains keypoints, include the entire segment
                        mask = np.logical_or(mask, instance_mask_np).astype(np.uint8)

            # Scale the mask to 255
            mask = mask * 255

            # Post-processing: remove small isolated regions and close gaps
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            cv2.imwrite(mask_file, mask)

        # Invert mask to get background
        bg_mask = cv2.bitwise_not(mask)

        # Apply mask to original frame
        bg_only = cv2.bitwise_and(frame, frame, mask=bg_mask)

        return bg_only, mask

    def process_video(self, input_path, force_reprocess=False):
        if force_reprocess:
            self.poses = []  # Clear existing poses if force_reprocess is True
        elif self.load_existing_data():
            print("Loaded existing pose data.")
            if os.path.exists(self.bg_video_path):
                print(f"Background video already exists at {self.bg_video_path}")
                return True
            else:
                print("Existing pose data found, but background video is missing. Reprocessing video...")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open the input video at {input_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.bg_video_path, fourcc, fps, (width, height))

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bg_only, _ = self.process_frame(frame, frame_num)
            out.write(bg_only)

            frame_num += 1

            if frame_num > 5:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Save poses to JSON file
        with open(self.pose_file, 'w') as f:
            json.dump(self.poses, f, indent=2)

        print(f"Processed {frame_num} frames")
        print(f"Masks saved in: {self.mask_dir}")
        print(f"Poses saved in: {self.pose_file}")
        print(f"Background video saved to: {self.bg_video_path}")
        return True

    def get_poses(self):
        return self.poses

    def get_bg_video_path(self):
        return self.bg_video_path