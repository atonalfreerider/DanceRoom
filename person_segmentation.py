import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import os
import glob


class DanceSegmentation:
    def __init__(self, input_path, output_dir, lead, follow):
        self.input_path = input_path
        self.output_dir = output_dir
        self.mask_dir = os.path.join(output_dir, 'masks')
        self.sam_predictor = None

    def process_video(self):
        self.segment_leads()
        print("Video processing complete.")

    def segment_leads(self):
        if not self.sam_predictor:
            sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
            self.sam_predictor = SamPredictor(sam)

        cap = cv2.VideoCapture(self.input_path)

        existing_masks = glob.glob(os.path.join(self.mask_dir, 'mask_lead_??????.png'))
        start_frame = len(existing_masks)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            lead_pose = self.lead.get(str(frame_count))
            follow_pose = self.follow.get(str(frame_count))

            if lead_pose:
                self.segment_person(frame, lead_pose, frame_count, 'lead')
            if follow_pose:
                self.segment_person(frame, follow_pose, frame_count, 'follow')

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames for segmentation")

        cap.release()

    def segment_person(self, frame, pose, frame_num, person_type):
        self.sam_predictor.set_image(frame)

        # Get bounding box from pose
        x_coords = [kp[0] for kp in pose if kp[3] > 0.5]  # Only consider high confidence keypoints
        y_coords = [kp[1] for kp in pose if kp[3] > 0.5]
        if not x_coords or not y_coords:
            return

        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        box = np.array([x1, y1, x2, y2])
        masks, _, _ = self.sam_predictor.predict(box=box, multimask_output=False)
        mask = masks[0].astype(np.uint8)

        mask_file = os.path.join(self.mask_dir, f'mask_{person_type}_{frame_num:06d}.png')
        cv2.imwrite(mask_file, mask * 255)

