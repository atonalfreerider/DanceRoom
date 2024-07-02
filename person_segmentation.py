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

    def load_existing_data(self):
        if os.path.exists(self.pose_file):
            with open(self.pose_file, 'r') as f:
                self.poses = json.load(f)
            return True
        return False

    def process_frame(self, frame, frame_num):
        mask_file = os.path.join(self.mask_dir, f'mask_{frame_num:06d}.png')

        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        else:
            # Segmentation
            seg_results = self.seg_model(frame)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for seg in seg_results:
                for instance_mask in seg.masks.data:
                    instance_mask_np = instance_mask.cpu().numpy()
                    mask = np.logical_or(mask, instance_mask_np).astype(np.uint8) * 255
            cv2.imwrite(mask_file, mask)

        if len(self.poses) <= frame_num:
            # Pose detection
            pose_results = self.pose_model(frame, classes=[0])
            frame_poses = []
            for r in pose_results:
                if r.keypoints is not None:
                    for person_keypoints in r.keypoints.data:
                        person_pose = person_keypoints.cpu().numpy().tolist()
                        frame_poses.append(person_pose)
            self.poses.append({"frame": frame_num, "poses": frame_poses})

        # Invert mask to get background
        bg_mask = cv2.bitwise_not(mask)

        # Apply mask to original frame
        bg_only = cv2.bitwise_and(frame, frame, mask=bg_mask)

        return bg_only, mask

    def process_video(self, input_path, output_path):
        if self.load_existing_data():
            print("Loaded existing pose data.")
            return

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bg_only, _ = self.process_frame(frame, frame_num)
            out.write(bg_only)

            frame_num += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Save poses to JSON file
        with open(self.pose_file, 'w') as f:
            json.dump(self.poses, f)

        print(f"Processed {frame_num} frames")
        print(f"Masks saved in: {self.mask_dir}")
        print(f"Poses saved in: {self.pose_file}")

    def get_poses(self):
        return self.poses