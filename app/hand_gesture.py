import cv2
import mediapipe as mp
import numpy as np

class HandGestureRecognizer:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, 0.0

        lm = result.multi_hand_landmarks[0].landmark

        def up(a, b): return lm[a].y < lm[b].y

        fingers = {
            "thumb": up(4, 2),
            "index": up(8, 6),
            "middle": up(12, 10),
            "ring": up(16, 14),
            "pinky": up(20, 18)
        }

                                   
        if all(fingers.values()):
            return "открытая ладонь", 0.9

        if not any(fingers.values()):
            return "сжатый кулак", 0.9

        if fingers["index"] and fingers["middle"] and not fingers["ring"]:
            return "два пальца (V)", 0.85

        if fingers["thumb"] and not fingers["index"]:
            return "большой палец вверх", 0.85

        if fingers["thumb"] and fingers["pinky"] and not fingers["index"]:
            return "жест телефон", 0.8

        return "жест рукой", 0.6
