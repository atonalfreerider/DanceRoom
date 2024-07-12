import os
import cv2
import numpy as np
from tqdm import tqdm


class Segmenter:
    def __init__(self, video_path, output_dir):
        # Create output directories
        self.video_path = video_path
        self.output_dir = output_dir
        self.figure_mask_dir = os.path.join(output_dir, "figure-masks")

        self.depth_dir = os.path.join(output_dir, 'depth')
        os.makedirs(self.figure_mask_dir, exist_ok=True)

    def process_video(self):
        # Open the video file
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Segmenting video based on depth")
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            depth_map = self.load_depth_map(int(frame_count))
            if depth_map is None:
                continue

            # Scale the depth map to match the frame size
            scaled_depth = cv2.resize(depth_map, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

            # Find the max and min depths for each row
            max_depths = np.max(scaled_depth, axis=1)
            min_depths = np.min(scaled_depth, axis=1)

            # Create a mask for rows where max_depth / min_depth >= 1.5
            valid_rows = max_depths / min_depths >= 1.5

            # Create the initial mask
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # For valid rows, mark pixels where depth < min_depth * 1.5
            row_indices, col_indices = np.where(
                valid_rows[:, np.newaxis] & (scaled_depth < min_depths[:, np.newaxis] * 1.5))
            mask[row_indices, col_indices] = 255

            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Save the masked frame
            mask_file = os.path.join(self.figure_mask_dir, f'{frame_count:06d}.png')
            cv2.imwrite(mask_file, masked_frame)

            pbar.update(1)

        # Release the video capture object
        cap.release()
        pbar.close()

        print("Video processing completed.")
        print("Video processing completed.")

    def load_depth_map(self, frame_num):
        depth_file = os.path.join(self.depth_dir, f'{frame_num:06d}.npz')
        if os.path.exists(depth_file):
            with np.load(depth_file) as data:
                # Try to get the first key in the archive
                keys = list(data.keys())
                if keys:
                    return data[keys[0]]
                else:
                    print(f"Warning: No data found in {depth_file}")
                    return None
        else:
            print(f"Warning: Depth file not found: {depth_file}")
            return None