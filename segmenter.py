import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter


# uses depth maps to isolate figures dancing in center of floor
class Segmenter:
    def __init__(self, video_path, output_dir):
        # Create output directories
        self.video_path = video_path
        self.output_dir = output_dir
        self.figure_mask_dir = os.path.join(output_dir, "figure-masks")

        self.depth_dir = os.path.join(output_dir, 'depth')
        os.makedirs(self.figure_mask_dir, exist_ok=True)

    def process_video(self):
        # return if mask dir already contains files
        if len(os.listdir(self.figure_mask_dir)) > 0:
            print("Figure masks already exist. Skipping video processing.")
            return

        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # List to store D values for each frame
        D_values = []

        # First pass: Calculate D values for each frame
        pbar = tqdm(total=total_frames, desc="Calculating D values")
        for frame_count in range(total_frames):
            depth_map = self.load_depth_map(int(frame_count))
            if depth_map is None:
                D_values.append(None)
                pbar.update(1)
                continue

            max_contrast = 0
            D = None
            for row in depth_map:
                sorted_row = np.sort(row)
                contrast = np.sum(sorted_row[::-1] - sorted_row)
                if contrast > max_contrast:
                    max_contrast = contrast
                    D = sorted_row[0]  # Closest depth value in the row with highest contrast

            D_values.append(D)
            pbar.update(1)

        pbar.close()

        # Remove None values and apply smoothing
        D_values_filtered = [d for d in D_values if d is not None]
        if len(D_values_filtered) > 0:
            smoothed_D = savgol_filter(D_values_filtered,
                                       window_length=min(51, len(D_values_filtered)),
                                       polyorder=3)
        else:
            print("Warning: No valid D values found.")
            return

        # Second pass: Create masks using smoothed D values
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pbar = tqdm(total=total_frames, desc="Creating masks")
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            depth_map = self.load_depth_map(int(frame_count))
            if depth_map is None or D_values[frame_count] is None:
                pbar.update(1)
                continue

            scaled_depth = cv2.resize(depth_map, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

            D = smoothed_D[frame_count]
            mask = ((scaled_depth >= D - 2) & (scaled_depth <= D + 2)).astype(np.uint8) * 255

            # Optional: Apply some morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Apply 5% expansion factor
            expansion_kernel_size = int(min(frame_width, frame_height) * 0.05)
            expansion_kernel = np.ones((expansion_kernel_size, expansion_kernel_size), np.uint8)
            mask = cv2.dilate(mask, expansion_kernel, iterations=1)

            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Save the masked frame
            mask_file = os.path.join(self.figure_mask_dir, f'{frame_count:06d}.png')
            cv2.imwrite(mask_file, masked_frame)

            pbar.update(1)

        cap.release()
        pbar.close()
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