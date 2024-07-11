import os
import json
import subprocess
import numpy as np
import cv2


class ColmapTracker:
    def __init__(self):
        pass

    def process_video(self, input_video, output_dir):
        sparse_dir = self.run_colmap(input_video, output_dir)
        camera_motions = self.extract_camera_motion(sparse_dir)

        # Save camera motions to JSON
        with open("camera_motions.json", "w") as f:
            json.dump(camera_motions, f, indent=2)

        # Create debug video
        self.create_debug_video(input_video, camera_motions, "debug_video.mp4")

    @staticmethod
    def run_colmap(video_path, output_dir):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract frames from video
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        subprocess.run(["ffmpeg", "-i", video_path, "-qscale:v", "1", f"{frames_dir}/frame_%06d.jpg"])

        # Run COLMAP feature extraction
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", f"{output_dir}/database.db",
            "--image_path", frames_dir,
            "--ImageReader.camera_model", "SIMPLE_RADIAL"
        ])

        # Run COLMAP sequential matching
        subprocess.run([
            "colmap", "sequential_matcher",
            "--database_path", f"{output_dir}/database.db"
        ])

        # Run COLMAP mapper
        sparse_dir = os.path.join(output_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        subprocess.run([
            "colmap", "mapper",
            "--database_path", f"{output_dir}/database.db",
            "--image_path", frames_dir,
            "--output_path", sparse_dir
        ])

        return sparse_dir

    def extract_camera_motion(self, sparse_dir):
        # Read cameras.txt
        cameras_file = os.path.join(sparse_dir, "0", "cameras.txt")
        with open(cameras_file, "r") as f:
            camera_lines = f.readlines()[3:]  # Skip header
        camera_params = [line.strip().split() for line in camera_lines]
        camera_id = int(camera_params[0][0])

        # Read images.txt
        images_file = os.path.join(sparse_dir, "0", "images.txt")
        with open(images_file, "r") as f:
            image_lines = f.readlines()[4:]  # Skip header
        image_lines = [line.strip() for line in image_lines if line.strip()]

        camera_motions = []
        for i in range(0, len(image_lines), 2):
            parts = image_lines[i].split()
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])

            # Convert quaternion to Euler angles
            angles = self.quaternion_to_euler(qw, qx, qy, qz)

            camera_motions.append({
                "frame": int(parts[8].split(".")[0].split("_")[-1]),
                "pan": angles[1],
                "tilt": angles[0],
                "roll": angles[2],
                "zoom": 1.0 / tz  # Approximating zoom as inverse of z-translation
            })

        return camera_motions

    @staticmethod
    def quaternion_to_euler(qw, qx, qy, qz):
        # Convert quaternion to Euler angles (in radians)
        t0 = 2 * (qw * qx + qy * qz)
        t1 = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(t0, t1)

        t2 = 2 * (qw * qy - qz * qx)
        t2 = 1 if t2 > 1 else -1 if t2 < -1 else t2
        pitch = np.arcsin(t2)

        t3 = 2 * (qw * qz + qx * qy)
        t4 = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(t3, t4)

        return pitch, yaw, roll

    @staticmethod
    def create_debug_video(video_path, camera_motions, output_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < len(camera_motions):
                motion = camera_motions[frame_idx]
                text = f"Frame: {motion['frame']}"
                text += f"\nPan: {motion['pan']:.2f}"
                text += f"\nTilt: {motion['tilt']:.2f}"
                text += f"\nRoll: {motion['roll']:.2f}"
                text += f"\nZoom: {motion['zoom']:.2f}"

                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()


