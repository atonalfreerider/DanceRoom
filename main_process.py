import os
import argparse

import yolo_pose
from segmenter import Segmenter
import room_tracker
from sam2 import Sam2
from dancer_tracker import DancerTracker


def main(input_video, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    sam2_tracker = Sam2()
    os.makedirs(output_dir + "/sam2", exist_ok=True)
    sam2_tracker.predict(input_video, output_dir + "/sam2")

    #segmenter = Segmenter(input_video, output_dir)
    #segmenter.process_video()

    #yoloPose = yolo_pose.YOLOPose(output_dir + "/figure-masks", output_dir + "/detections.json")
    #yoloPose.detect_poses()

    #room_tracker.room_tracker(input_video, output_dir)
    #room_tracker.debug_video(input_video, output_dir, output_dir + "/deltas.json")

    # Step 1: Segment people, create background-only video, and save masks and poses
    #dancer_tracker = DancerTracker(input_video, output_dir)
    #dancer_tracker.process_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.output_dir)