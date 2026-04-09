import cv2
from detector import PersonDetector
from pose import PoseEstimator
from intent_predictor import IntentPredictor
from draw import draw_text

                                                         
cap = cv2.VideoCapture(0)

detector = PersonDetector()
pose_estimator = PoseEstimator()
action_model = IntentPredictor()

                                                
WINDOW_NAME = "Task A: Motion Classification"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

                                                         
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame, _ = frame.shape
    boxes = detector.detect(frame) or []
    if boxes:
                                                                                     
        boxes = [max(boxes, key=lambda b: max(0, (b[2] - b[0])) * max(0, (b[3] - b[1])))]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        person = frame[y1:y2, x1:x2]
        if person.size < 500:
            continue

                                                        
        pose = pose_estimator.estimate(person)
        if pose is None:
            continue

                                                                        
        pose_estimator.draw_landmarks(person, pose, draw_labels=True)
        action_model.update(box, pose)
        action, action_prob = action_model.detect_action(pose)
        binary_scores = action_model.get_binary_scores()
        trajectory = action_model.get_recent_trajectory(max_points=90)

                                                             
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for i in range(1, len(trajectory)):
            shade = int(80 + 170 * (i / max(1, len(trajectory) - 1)))
            color = (0, shade, 255 - shade // 3)
            cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)

        if trajectory:
            cv2.circle(frame, trajectory[-1], 5, (0, 0, 255), -1)

        frame = draw_text(
            frame,
            f"действие A: {action} ({int(action_prob * 100)}%)",
            (x1, max(35, y1 - 35)),
            34
        )

        frame = draw_text(
            frame,
            (
                f"bin: статика {int(binary_scores['статика']*100)}% | "
                f"шаг {int(binary_scores['шаг']*100)}% | "
                f"присед {int(binary_scores['присед']*100)}%"
            ),
            (x1, min(h_frame - 50, y2 + 25)),
            22
        )
        frame = draw_text(
            frame,
            (
                f"bin: прыжок {int(binary_scores['прыжок']*100)}% | "
                f"мах рукой {int(binary_scores['мах рукой']*100)}%"
            ),
            (x1, min(h_frame - 20, y2 + 52)),
            22
        )

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break

                                                      
cap.release()
cv2.destroyAllWindows()
