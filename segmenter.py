import os
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import cv2
import numpy as np
from tqdm import tqdm


class Segmenter:
    def __init__(self, video_path, output_dir):
        # Create output directories
        self.video_path = video_path
        self.output_dir = output_dir
        self.background_dir = os.path.join(output_dir, "background")
        os.makedirs(self.background_dir, exist_ok=True)

    def process_video(self):
        # Create a FastSAM model
        model = FastSAM("FastSAM-x.pt")

        # Open the video file
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frame_count = 0

        # Create a directory to save detected object images
        objects_dir = os.path.join(self.output_dir, "detected_objects")
        os.makedirs(objects_dir, exist_ok=True)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing frames")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pbar.update(1)

            # Run inference on the frame
            results = model(frame, device="cpu", retina_masks=True, imgsz=frame_width, conf=0.4, iou=0.9)

            # Process results
            if len(results) > 0:
                result = results[0]  # Assuming we're processing one image at a time

                # Print detected classes and confidence scores
                if hasattr(result, 'boxes') and result.boxes is not None:
                    print(f"Frame {frame_count} detections:")
                    for box in result.boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        print(f"  Class: {class_id}, Confidence: {confidence:.2f}")

                # Save images for all detected objects
                if hasattr(result, 'masks') and result.masks is not None:
                    for i, mask_tensor in enumerate(result.masks.data):
                        mask = mask_tensor.cpu().numpy().astype(np.uint8)

                        # Extract the object from the frame
                        object_image = cv2.bitwise_and(frame, frame, mask=mask)

                        # Save the object image
                        object_path = os.path.join(objects_dir, f"frame_{frame_count:04d}_object_{i + 1:02d}.jpg")
                        cv2.imwrite(object_path, object_image)

            else:
                print(f"No detections in frame {frame_count}")

            print(f"Processed frame {frame_count}")

        # Release the video capture object
        cap.release()
        pbar.close()

        print("Video processing completed.")
        print("Video processing completed.")