import os
import cv2


class DebugVideo:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir

    def generate_debug_video(self):
        debug_video_path = os.path.join(self.output_dir, 'debug_video.mp4')
        cap = cv2.VideoCapture(self.input_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(debug_video_path, fourcc, fps, (frame_width, frame_height))

        # Load tracked sequences
        lead_track = self.load_json(self.lead_file)
        follow_track = self.load_json(self.follow_file)

        # Restructure detections for efficient access
        detections_by_frame = {int(frame_id): detections for frame_id, detections in self.detections.items()}

        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw all tracked poses for the current frame
            detections_in_frame = detections_by_frame.get(frame_count, [])
            for detection in detections_in_frame:
                pose = detection.get('keypoints')
                gender = detection.get('gender')
                track_id = detection.get('id')
                bbox = detection.get('bbox')

                if pose:
                    color = (125, 125, 125)
                    if gender['gender'] == 'female':
                        color = (100, 100, 255)
                    elif gender['gender'] == 'male':
                        color = (255, 0, 0)
                    self.draw_pose(frame, pose, color)

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Put track ID text
                    if track_id is not None:
                        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw lead and follow
            lead_pose = lead_track.get(str(frame_count))
            if lead_pose:
                self.draw_pose(frame, lead_pose, (0, 0, 255), is_lead_or_follow=True)  # Red for lead
                cv2.putText(frame, "LEAD", (int(lead_pose[0][0]), int(lead_pose[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            follow_pose = follow_track.get(str(frame_count))
            if follow_pose:
                self.draw_pose(frame, follow_pose, (255, 192, 203), is_lead_or_follow=True)  # Pink for follow
                cv2.putText(frame, "FOLLOW", (int(follow_pose[0][0]), int(follow_pose[0][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 192, 203), 2)

            out.write(frame)

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames for debug video")

        cap.release()
        out.release()

    @staticmethod
    def draw_pose(image, pose, color, is_lead_or_follow=False):
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]

        overlay = np.zeros_like(image, dtype=np.uint8)

        def get_point_and_conf(kp):
            if len(kp) == 4 and (kp[0] != 0 or kp[1] != 0):
                return (int(kp[0]), int(kp[1])), kp[3]  # x, y, z, conf
            elif len(kp) == 3 and (kp[0] != 0 or kp[1] != 0):
                return (int(kp[0]), int(kp[1])), kp[2]  # x, y, conf
            return None, 0.0

        line_thickness = 3 if is_lead_or_follow else 1

        for connection in connections:
            if len(pose) > max(connection):
                start_point, start_conf = get_point_and_conf(pose[connection[0]])
                end_point, end_conf = get_point_and_conf(pose[connection[1]])

                if start_point is not None and end_point is not None:
                    avg_conf = (start_conf + end_conf) / 2
                    color_with_alpha = tuple(int(c * avg_conf) for c in color)
                    cv2.line(overlay, start_point, end_point, color_with_alpha, line_thickness)

        for point in pose:
            pt, conf = get_point_and_conf(point)
            if pt is not None:
                color_with_alpha = tuple(int(c * conf) for c in color)
                cv2.circle(overlay, pt, 3, color_with_alpha, -1)

        cv2.add(image, overlay, image)