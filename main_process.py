import os
import argparse

import room_tracker
from person_segmentation import DanceSegmentation


def main(input_video, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    deltas = room_tracker.room_tracker(input_video, output_dir + "/deltas.json", output_dir + "/debug-points.mp4")
    #room_tracker.debug_render(input_video, deltas, output_dir + "/delta-video.mp4")

    # Step 1: Segment people, create background-only video, and save masks and poses
    #segmentation = DanceSegmentation(input_video, output_dir)
    #segmentation.process_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.input_video, args.output_dir)