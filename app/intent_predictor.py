import math
from collections import deque

import numpy as np


class IntentPredictor:
    """Вариант A: one-vs-rest (5 бинарных классификаторов действий)."""

    REQUIRED_CLASSES = ("статика", "шаг", "присед", "прыжок", "мах рукой")

    def __init__(self):
                                                                   
        self.centers = deque(maxlen=180)
        self.heights = deque(maxlen=180)

                                                                         
        self.vx_hist = deque(maxlen=60)
        self.vy_hist = deque(maxlen=60)                     
        self.speed_hist = deque(maxlen=60)

                       
        self.left_wrist_y_hist = deque(maxlen=60)
        self.right_wrist_y_hist = deque(maxlen=60)
        self.hand_motion_hist = deque(maxlen=60)
        self.hands_up_hist = deque(maxlen=60)

                               
        self.ankle_delta_hist = deque(maxlen=60)
        self.ankle_mean_hist = deque(maxlen=60)
        self.knee_angle_hist = deque(maxlen=60)
        self.hip_ratio_hist = deque(maxlen=60)

        self.prev_left_wrist = None
        self.prev_right_wrist = None

                                                  
        self.binary_ema = {cls: 0.0 for cls in self.REQUIRED_CLASSES}
        self.latest_scores = {cls: 0.0 for cls in self.REQUIRED_CLASSES}

        self.last_action = "статика"
        self.action_hist = deque(maxlen=10)
        self.jump_cooldown = 0

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _std(values):
        return float(np.std(values)) if len(values) > 1 else 0.0

    @staticmethod
    def _ptp(values):
        return float(np.ptp(values)) if len(values) > 1 else 0.0

    @staticmethod
    def _sign_changes(values, eps=1e-4):
        if len(values) < 2:
            return 0

        signs = []
        for v in values:
            if v > eps:
                signs.append(1)
            elif v < -eps:
                signs.append(-1)

        if len(signs) < 2:
            return 0

        changes = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                changes += 1
        return changes

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

    def update(self, bbox, pose):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bbox_h = max(y2 - y1, 1.0)

        if self.centers:
            px, py = self.centers[-1]
            dx = (cx - px) / bbox_h
            dy = (cy - py) / bbox_h
            self.vx_hist.append(dx)
            self.vy_hist.append(-dy)             
            self.speed_hist.append(math.hypot(dx, dy))

        self.centers.append((cx, cy))
        self.heights.append(bbox_h)

        ls, rs = pose[11], pose[12]
        lw, rw = pose[15], pose[16]
        lh, rh = pose[23], pose[24]
        lk, rk = pose[25], pose[26]
        la, ra = pose[27], pose[28]

        shoulder_width = max(abs(ls.x - rs.x), 0.05)

        if self.prev_left_wrist is not None and self.prev_right_wrist is not None:
            l_disp = math.hypot(lw.x - self.prev_left_wrist[0], lw.y - self.prev_left_wrist[1])
            r_disp = math.hypot(rw.x - self.prev_right_wrist[0], rw.y - self.prev_right_wrist[1])
            self.hand_motion_hist.append(max(l_disp, r_disp) / shoulder_width)

        self.prev_left_wrist = (lw.x, lw.y)
        self.prev_right_wrist = (rw.x, rw.y)

        self.left_wrist_y_hist.append(lw.y)
        self.right_wrist_y_hist.append(rw.y)
        self.hands_up_hist.append(float(lw.y < ls.y or rw.y < rs.y))

        self.ankle_delta_hist.append(la.y - ra.y)
        self.ankle_mean_hist.append((la.y + ra.y) / 2.0)

        knee_l = self._angle(lh, lk, la)
        knee_r = self._angle(rh, rk, ra)
        self.knee_angle_hist.append(min(knee_l, knee_r))

        shoulder_mid = (ls.y + rs.y) / 2.0
        hip_mid = (lh.y + rh.y) / 2.0
        ankle_mid = (la.y + ra.y) / 2.0
        denom = max(ankle_mid - shoulder_mid, 0.08)
        self.hip_ratio_hist.append((hip_mid - shoulder_mid) / denom)

    def _binary_scores(self):
        vx_recent = list(self.vx_hist)[-16:]
        vy_recent = list(self.vy_hist)[-16:]
        speed_recent = list(self.speed_hist)[-16:]
        ankle_delta_recent = list(self.ankle_delta_hist)[-18:]
        ankle_mean_recent = list(self.ankle_mean_hist)[-18:]
        knee_recent = list(self.knee_angle_hist)[-12:]
        hip_recent = list(self.hip_ratio_hist)[-12:]
        hand_recent = list(self.hand_motion_hist)[-12:]
        hands_up_recent = list(self.hands_up_hist)[-12:]
        lw_y_recent = list(self.left_wrist_y_hist)[-14:]
        rw_y_recent = list(self.right_wrist_y_hist)[-14:]

        speed_mean = self._mean(speed_recent)
        horiz_mean = self._mean([abs(v) for v in vx_recent])
        vy_up = max(vy_recent) if vy_recent else 0.0
        vy_down = min(vy_recent) if vy_recent else 0.0
        vy_span = self._ptp(vy_recent)

        hand_motion = self._mean(hand_recent)
        hands_up_ratio = self._mean(hands_up_recent)
        wrist_amp = max(self._ptp(lw_y_recent), self._ptp(rw_y_recent))

        ankle_swap = self._std(ankle_delta_recent)
        ankle_changes = self._sign_changes(ankle_delta_recent, eps=0.012)
        ankle_amp = self._ptp(ankle_mean_recent)

        knee_now = knee_recent[-1] if knee_recent else 170.0
        knee_mean = self._mean(knee_recent)
        hip_now = hip_recent[-1] if hip_recent else 0.55
        low_pose_ratio = self._mean([1.0 if k < 145.0 else 0.0 for k in knee_recent])

                                                                         
        squat_depth_now = self._clip01((148.0 - knee_now) / 40.0)
        squat_depth_mean = self._clip01((150.0 - knee_mean) / 36.0)
        squat_hip = self._clip01((hip_now - 0.57) / 0.16)
        squat_low_motion = self._clip01((0.07 - horiz_mean) / 0.07)
        squat_stable_vertical = 1.0 - self._clip01((ankle_amp - 0.07) / 0.10)
        squat_score = (
            0.30 * squat_depth_now +
            0.24 * squat_depth_mean +
            0.22 * squat_hip +
            0.14 * low_pose_ratio +
            0.05 * squat_low_motion +
            0.05 * squat_stable_vertical
        )
        if ankle_changes >= 5 and horiz_mean > 0.028:
            squat_score *= 0.75

                                                                                     
        takeoff = self._clip01((vy_up - 0.07) / 0.12)
        landing = self._clip01((abs(vy_down) - 0.07) / 0.12)
        impulse = takeoff * landing
        flight = self._clip01((ankle_amp - 0.08) / 0.10)
        knee_extended = self._clip01((knee_mean - 133.0) / 28.0)
        jump_score = (
            0.45 * impulse +
            0.22 * flight +
            0.18 * knee_extended +
            0.10 * self._clip01((vy_span - 0.10) / 0.15) +
            0.05 * self._clip01((speed_mean - 0.03) / 0.07)
        )
        if impulse < 0.14:
            jump_score *= 0.55
        if knee_now < 122.0 and hip_now > 0.64:
            jump_score *= 0.55
        if self.jump_cooldown > 0:
            jump_score *= 0.65

                                                                              
        wave_score = (
            0.40 * self._clip01((hand_motion - 0.045) / 0.12) +
            0.25 * self._clip01((wrist_amp - 0.08) / 0.16) +
            0.25 * self._clip01((hands_up_ratio - 0.20) / 0.60) +
            0.10 * self._clip01((0.05 - ankle_swap) / 0.05)
        )

                                                                    
        step_motion = self._clip01((horiz_mean - 0.016) / 0.07)
        step_leg_alt = self._clip01((ankle_swap - 0.018) / 0.07)
        step_leg_phase = self._clip01((ankle_changes - 2.0) / 4.0)
        step_score = (
            0.38 * step_motion +
            0.30 * step_leg_alt +
            0.20 * step_leg_phase +
            0.07 * self._clip01((knee_now - 118.0) / 30.0) +
            0.05 * self._clip01((speed_mean - 0.020) / 0.060)
        )
                                                 
        step_score *= (1.0 - 0.30 * self._clip01((hip_now - 0.64) / 0.12))
        step_score *= (1.0 - 0.35 * self._clip01((ankle_amp - 0.11) / 0.10))

                                                       
        static_score = (
            0.40 * self._clip01((0.028 - speed_mean) / 0.028) +
            0.25 * self._clip01((0.030 - hand_motion) / 0.030) +
            0.20 * self._clip01((0.020 - ankle_swap) / 0.020) +
            0.15 * self._clip01((knee_now - 145.0) / 25.0)
        )
        static_score *= (1.0 - 0.35 * self._clip01((ankle_amp - 0.05) / 0.08))

        return {
            "статика": self._clip01(static_score),
            "шаг": self._clip01(step_score),
            "присед": self._clip01(squat_score),
            "прыжок": self._clip01(jump_score),
            "мах рукой": self._clip01(wave_score),
        }

    def detect_action(self, pose):
        if len(self.centers) < 6:
            self.latest_scores = {k: 0.0 for k in self.REQUIRED_CLASSES}
            self.latest_scores["статика"] = 0.55
            self.binary_ema = self.latest_scores.copy()
            return "статика", 0.55

        raw_scores = self._binary_scores()

        for action, score in raw_scores.items():
            self.binary_ema[action] = 0.70 * self.binary_ema[action] + 0.30 * score

        self.latest_scores = dict(self.binary_ema)

        jump_score = self.binary_ema["прыжок"]
        if jump_score > 0.78 and self.jump_cooldown == 0:
            self.jump_cooldown = 8
        elif self.jump_cooldown > 0:
            self.jump_cooldown -= 1

        action = max(self.binary_ema, key=self.binary_ema.get)
        conf = self.binary_ema[action]

                                                                                                  
        if action == "прыжок" and conf < 0.58:
            alt = {k: v for k, v in self.binary_ema.items() if k != "прыжок"}
            action = max(alt, key=alt.get)
            conf = alt[action]

                                                                  
        if conf < 0.43:
            action = "статика"
            conf = max(conf, self.binary_ema["статика"])

                                      
        if action != self.last_action:
            margin = 0.08 if action == "прыжок" else 0.05
            if conf < self.binary_ema.get(self.last_action, 0.0) + margin:
                action = self.last_action
                conf = self.binary_ema.get(action, conf * 0.9)

                                                                   
        self.action_hist.append(action)
        if self.action_hist:
            majority = max(set(self.action_hist), key=self.action_hist.count)
            if majority != action and self.binary_ema[majority] + 0.03 >= self.binary_ema[action]:
                action = majority
                conf = self.binary_ema[action]

        self.last_action = action
        return action, round(min(max(conf, 0.01), 0.95), 2)

    def get_binary_scores(self):
        return {k: round(v, 3) for k, v in self.latest_scores.items()}

    def get_recent_trajectory(self, max_points=90):
        pts = list(self.centers)[-max_points:]
        return [(int(x), int(y)) for x, y in pts]

    def detect_intent(self, pose):
        return "не используется", 0.0
