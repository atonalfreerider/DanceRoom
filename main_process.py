import os
import argparse

import yolo_pose
from segmenter import Segmenter
import room_tracker
from dancer_tracker import DancerTracker


def main(input_video, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    yoloPose = yolo_pose.YOLOPose(input_video, output_dir + "/detections.json")
    yoloPose.detect_poses()
    #segmenter = Segmenter(input_video, output_dir)
    #segmenter.process_video()

    room_tracker.room_tracker(input_video, output_dir + "/deltas.json", output_dir + "/debug-points.mp4", output_dir + "/detections.json")

    # Step 1: Segment people, create background-only video, and save masks and poses
    #dancer_tracker = DancerTracker(input_video, output_dir)
    #dancer_tracker.process_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.output_dir)