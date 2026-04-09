import cv2
import os
import shutil
import subprocess
from detector import PersonDetector
from pose import PoseEstimator
from intent_predictor import IntentPredictor
from draw import draw_text

detector = PersonDetector()
pose_estimator = PoseEstimator()
action_model = IntentPredictor()


def _transcode_for_telegram(video_path, target_fps=None):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not os.path.exists(video_path):
        return False

    tmp_path = f"{video_path}.h264.tmp.mp4"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
    ]
    if target_fps:
        cmd += ["-r", f"{float(target_fps):.2f}"]
    cmd.append(tmp_path)

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0 or not os.path.exists(tmp_path):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

        if os.path.getsize(tmp_path) < 1024:
            os.remove(tmp_path)
            return False

        os.replace(tmp_path, video_path)
        return True
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

def process_frame(frame, model=None):
    model = model or action_model
    h, w, _ = frame.shape
    boxes = detector.detect(frame) or []
    if boxes:
        boxes = [max(boxes, key=lambda b: max(0, (b[2] - b[0])) * max(0, (b[3] - b[1])))]

    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        person = frame[y1:y2, x1:x2]
        if person.size == 0:
            continue

        pose = pose_estimator.estimate(person)
        if pose is None:
            continue

                                                                     
        pose_estimator.draw_landmarks(person, pose, draw_labels=True)
        model.update(box, pose)
        action, action_prob = model.detect_action(pose)
        binary_scores = model.get_binary_scores()
        trajectory = model.get_recent_trajectory(max_points=90)

                                     
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        for i in range(1, len(trajectory)):
            shade = int(80 + 170 * (i / max(1, len(trajectory) - 1)))
            color = (0, shade, 255 - shade // 3)
            cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)

        if trajectory:
            cv2.circle(frame, trajectory[-1], 5, (0, 0, 255), -1)

        frame = draw_text(
            frame,
            f"действие A: {action} ({int(action_prob*100)}%)",
            (x1, max(25, y1-35)),
            28
        )
        frame = draw_text(
            frame,
            (
                f"bin: статика {int(binary_scores['статика']*100)}% | "
                f"шаг {int(binary_scores['шаг']*100)}% | "
                f"присед {int(binary_scores['присед']*100)}%"
            ),
            (x1, min(h - 44, y2 + 23)),
            19
        )
        frame = draw_text(
            frame,
            (
                f"bin: прыжок {int(binary_scores['прыжок']*100)}% | "
                f"мах рукой {int(binary_scores['мах рукой']*100)}%"
            ),
            (x1, min(h - 18, y2 + 45)),
            19
        )

    return frame


def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    model = IntentPredictor()
    out = process_frame(img, model=model)
    cv2.imwrite(output_path, out)


def process_video(input_path, output_path, output_max_width=None, target_fps=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
                                                                                                    
        shutil.copyfile(input_path, output_path)
        return

    model = IntentPredictor()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                                                                                
                                                                           
    if raw_fps < 5.0 or raw_fps > 120.0:
        fps = 25.0
    else:
        fps = raw_fps

    scale = 1.0
    if output_max_width and w > output_max_width:
        scale = output_max_width / float(w)

    out_w = max(2, int(round(w * scale)))
    out_h = max(2, int(round(h * scale)))
    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1

    out_fps = min(float(target_fps), fps) if target_fps else fps
    out_fps = max(1.0, out_fps)
    frame_step = max(1, int(round(fps / out_fps)))
                                                     
    if frame_step > 5:
        frame_step = 1
        out_fps = fps

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (out_w, out_h)
    )
    if not out.isOpened():
        cap.release()
        shutil.copyfile(input_path, output_path)
        return

    frame_idx = 0
    written_frames = 0
    prev_original_gray = None
    prev_output_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        if scale != 1.0 or frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        original_frame = frame.copy()
        try:
            processed = process_frame(frame, model=model)
        except Exception:
            processed = original_frame

        if processed is None:
            processed = original_frame

        if processed.shape[:2] != (out_h, out_w):
            processed = cv2.resize(processed, (out_w, out_h), interpolation=cv2.INTER_AREA)

                                                                                        
        if float(processed.mean()) < 2.0 and float(original_frame.mean()) > 10.0:
            processed = original_frame

                                              
                                                                                      
        original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        output_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        if prev_original_gray is not None and prev_output_gray is not None:
            in_diff = float(cv2.absdiff(original_gray, prev_original_gray).mean())
            out_diff = float(cv2.absdiff(output_gray, prev_output_gray).mean())
            if in_diff > 1.2 and out_diff < 0.12:
                processed = original_frame
                output_gray = original_gray

        prev_original_gray = original_gray
        prev_output_gray = output_gray

        out.write(processed)
        written_frames += 1
        frame_idx += 1

    cap.release()
    out.release()

                                                                              
    if written_frames == 0:
        shutil.copyfile(input_path, output_path)
        return

                                                                  
    _transcode_for_telegram(output_path, target_fps=out_fps)
