import os
import warnings

                                                                         
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
                                                        
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

                                                                                           
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)

import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False
        )
        self.face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,                   
            min_detection_confidence=0.4                                        
        )
        self.label_points = {
            0: "nose",
            11: "L_sh",
            12: "R_sh",
            13: "L_elb",
            14: "R_elb",
            15: "L_wr",
            16: "R_wr",
            23: "L_hip",
            24: "R_hip",
            25: "L_knee",
            26: "R_knee",
            27: "L_ank",
            28: "R_ank",
        }

    def estimate(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark

    def detect_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face.process(rgb)
        if not result.detections:
            return None
        det = result.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(max(0, bbox.xmin) * w)
        y1 = int(max(0, bbox.ymin) * h)
        x2 = int(min(1, bbox.xmin + bbox.width) * w)
        y2 = int(min(1, bbox.ymin + bbox.height) * h)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def draw_landmarks(self, frame, landmarks, draw_labels=True, visibility_thr=0.45):
        if landmarks is None:
            return frame

        h, w, _ = frame.shape

        for p1, p2 in self.mp_pose.POSE_CONNECTIONS:
            lm1, lm2 = landmarks[p1], landmarks[p2]
            if lm1.visibility < visibility_thr or lm2.visibility < visibility_thr:
                continue

            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (120, 255, 120), 2)

        for idx, lm in enumerate(landmarks):
            if lm.visibility < visibility_thr:
                continue
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 180, 255), -1)

            if draw_labels and idx in self.label_points:
                cv2.putText(
                    frame,
                    self.label_points[idx],
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return frame
