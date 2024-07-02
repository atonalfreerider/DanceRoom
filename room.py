import cv2
import numpy as np
import math
import os


def detect_lines_and_axes(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open the video at {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, 'output_video.mp4'), fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Calculate 2% padding
        pad_x = int(width * 0.01)
        pad_y = int(height * 0.01)

        # Create padding mask
        padding_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        padding_mask[pad_y:height - pad_y, pad_x:width - pad_x] = 255

        # Upper half mask (for vertical lines)
        upper_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        upper_mask[pad_y:height // 2, pad_x:width - pad_x] = 255

        # Lower half mask (for floor lines)
        lower_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        lower_mask[height // 2:height - pad_y, pad_x:width - pad_x] = 255

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Apply padding mask to edges
        edges = cv2.bitwise_and(edges, edges, mask=padding_mask)

        # Upper half processing (vertical lines)
        upper_edges = cv2.bitwise_and(edges, edges, mask=upper_mask)
        upper_lines = cv2.HoughLinesP(upper_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        vertical_lines = []
        if upper_lines is not None:
            for line in upper_lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:  # Perfectly vertical line
                    angle = 90
                else:
                    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if abs(angle - 90) < 20:
                    vertical_lines.append((x1, y1, x2, y2, angle))

        # Calculate roll angle based on vertical lines
        if vertical_lines:
            angles = [90 - angle for _, _, _, _, angle in vertical_lines]
            roll_angle = np.median(angles)
        else:
            roll_angle = 0

        # Draw vertical lines (green) in upper half
        for x1, y1, x2, y2, _ in vertical_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Lower half processing (floor lines)
        lower_edges = cv2.bitwise_and(edges, edges, mask=lower_mask)
        lower_lines = cv2.HoughLinesP(lower_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        floor_lines = []
        if lower_lines is not None:
            for line in lower_lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
                floor_lines.append((x1, y1, x2, y2, angle))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Determine predominant x and y axes
        if floor_lines:
            angles = np.array([angle for _, _, _, _, angle in floor_lines])
            x_axis_angle = np.median(angles[np.abs(angles) < 45])
            y_axis_angle = np.median(angles[np.abs(angles - 90) < 45])
        else:
            x_axis_angle, y_axis_angle = 0, 90

        # Draw predominant x and y axes
        center_x, center_y = width // 2, height // 2
        line_length = 100

        # X-axis (blue)
        end_x = int(center_x + line_length * math.cos(math.radians(x_axis_angle)))
        end_y = int(center_y + line_length * math.sin(math.radians(x_axis_angle)))
        cv2.line(frame, (center_x, center_y), (end_x, end_y), (255, 0, 0), 2)

        # Y-axis (yellow)
        end_x = int(center_x + line_length * math.cos(math.radians(y_axis_angle)))
        end_y = int(center_y + line_length * math.sin(math.radians(y_axis_angle)))
        cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2)

        # Display roll angle and axes angles
        cv2.putText(frame, f"Roll: {roll_angle:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"X-axis: {x_axis_angle:.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Y-axis: {y_axis_angle:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw padding boundary (optional, for visualization)
        cv2.rectangle(frame, (pad_x, pad_y), (width - pad_x, height - pad_y), (255, 255, 0), 2)

        out.write(frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved to: {os.path.join(output_dir, 'output_video.mp4')}")


# Usage
video_path = '/home/john/Downloads/larissa-kadu-counter-demo/01-Demo-Ali.mp4'  # Replace with your actual video path
output_dir = '/home/john/Desktop/out'  # Replace with your desired output directory

detect_lines_and_axes(video_path, output_dir)