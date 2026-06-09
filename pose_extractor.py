from typing import Optional
import mediapipe as mp
import cv2
import numpy as np


# 33 landmarks x (x, y, z, visibility) = 132 values per frame
LANDMARK_DIM = 33 * 4


class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame: cv2.Mat, draw: bool = True) -> Optional[np.ndarray]:
        """Run pose inference ONCE. Optionally draw the skeleton on the frame.
        Returns a flat array of 132 floats, or None if no pose was detected."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None
        if draw:
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        lm = results.pose_landmarks.landmark
        return np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32).flatten()

    def close(self):
        self.pose.close()
