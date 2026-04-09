import math
from collections import Counter, deque

import numpy as np


class IntentPredictor:
    """Классификатор варианта A финального проекта."""

    REQUIRED_CLASSES = ("статика", "шаг", "присед", "прыжок", "мах рукой")

    def __init__(self):
                                                                                   
        self.centers = deque(maxlen=150)
        self.body_speed_hist = deque(maxlen=20)
        self.vertical_vel_hist = deque(maxlen=12)
        self.hand_motion_hist = deque(maxlen=15)
        self.ankle_delta_hist = deque(maxlen=15)
        self.knee_angle_hist = deque(maxlen=15)
        self.hands_up_hist = deque(maxlen=15)
        self.action_hist = deque(maxlen=8)

        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.last_action = "статика"

    @staticmethod
    def _angle(a, b, c):
        ab = (a.x - b.x, a.y - b.y)
        cb = (c.x - b.x, c.y - b.y)
        dot = ab[0] * cb[0] + ab[1] * cb[1]
        ab_len = math.hypot(*ab)
        cb_len = math.hypot(*cb)
        denom = max(ab_len * cb_len, 1e-7)
        value = max(-1.0, min(1.0, dot / denom))
        return math.degrees(math.acos(value))

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _std(values):
        return float(np.std(values)) if len(values) > 1 else 0.0

    def update(self, bbox, pose):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if self.centers:
            px, py = self.centers[-1]
            self.body_speed_hist.append(math.hypot(cx - px, cy - py))
            self.vertical_vel_hist.append(py - cy)                                           

        self.centers.append((cx, cy))

        ls, rs = pose[11], pose[12]
        lw, rw = pose[15], pose[16]
        lk, rk = pose[25], pose[26]
        la, ra = pose[27], pose[28]
        lh, rh = pose[23], pose[24]

        shoulder_width = max(abs(ls.x - rs.x), 1e-4)

        if self.prev_left_wrist is not None and self.prev_right_wrist is not None:
            l_disp = math.hypot(lw.x - self.prev_left_wrist[0], lw.y - self.prev_left_wrist[1])
            r_disp = math.hypot(rw.x - self.prev_right_wrist[0], rw.y - self.prev_right_wrist[1])
            hand_motion = max(l_disp, r_disp) / shoulder_width
            self.hand_motion_hist.append(hand_motion)

        self.prev_left_wrist = (lw.x, lw.y)
        self.prev_right_wrist = (rw.x, rw.y)

        self.ankle_delta_hist.append(la.y - ra.y)
        knee_l = self._angle(lh, lk, la)
        knee_r = self._angle(rh, rk, ra)
        self.knee_angle_hist.append(min(knee_l, knee_r))

        hands_up = float(lw.y < ls.y or rw.y < rs.y)
        self.hands_up_hist.append(hands_up)

    def detect_action(self, pose):
        if len(self.centers) < 2:
            return "статика", 0.55

        recent_centers = list(self.centers)[-20:]
        speed_mean = self._mean(self.body_speed_hist)
        speed_last = self.body_speed_hist[-1] if self.body_speed_hist else 0.0
        vertical_amp = (
            max(c[1] for c in recent_centers) - min(c[1] for c in recent_centers)
            if len(recent_centers) > 1
            else 0.0
        )
        hand_motion = self._mean(self.hand_motion_hist)
        ankle_swap = self._std(self.ankle_delta_hist)
        knee_angle = self._mean(self.knee_angle_hist)
        hands_up_ratio = self._mean(self.hands_up_hist)

        recent_vel = list(self.vertical_vel_hist)[-6:]
        oscillates_vertically = bool(
            len(recent_vel) >= 3 and max(recent_vel) > 2.0 and min(recent_vel) < -2.0
        )

        scores = {}

                    
        if speed_mean < 1.2 and vertical_amp < 12 and hand_motion < 0.025:
            score = 0.72 + max(0.0, (1.2 - speed_mean) * 0.15)
            scores["статика"] = min(score, 0.94)

                                                     
        if 1.4 <= speed_mean <= 8.0 and ankle_swap > 0.02 and vertical_amp < 26:
            score = 0.7 + min(0.2, ankle_swap * 2.5)
            scores["шаг"] = min(score, 0.93)

                                                                    
        if knee_angle < 145 and speed_mean < 4.0:
            depth_bonus = min(0.2, max(0.0, (145 - knee_angle) / 80))
            scores["присед"] = min(0.73 + depth_bonus, 0.94)

                                                                              
        if vertical_amp > 20 and oscillates_vertically and speed_last > 1.0:
            jump_bonus = min(0.2, (vertical_amp - 20) / 80)
            scores["прыжок"] = min(0.75 + jump_bonus, 0.95)

                                                                         
        if hands_up_ratio > 0.35 and hand_motion > 0.045 and speed_mean < 5.5:
            wave_bonus = min(0.2, (hand_motion - 0.045) * 5.0)
            scores["мах рукой"] = min(0.72 + wave_bonus, 0.94)

        if scores:
            action = max(scores, key=scores.get)
            conf = scores[action]
        else:
            action = self.last_action
            conf = 0.45

        self.action_hist.append(action)
        smooth_action = Counter(self.action_hist).most_common(1)[0][0]

        if smooth_action != action:
            conf *= 0.9

        self.last_action = smooth_action
        return smooth_action, round(min(conf, 0.95), 2)

    def get_recent_trajectory(self, max_points=90):
        pts = list(self.centers)[-max_points:]
        return [(int(x), int(y)) for x, y in pts]

    def detect_intent(self, pose):
                                                                 
        return "не используется", 0.0
