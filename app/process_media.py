import math
import os
import shutil
import subprocess

import cv2
import numpy as np

from detector import PersonDetector
from draw import draw_text
from intent_predictor import IntentPredictor
from pose import PoseEstimator

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


def _extract_skeleton_points_3d(pose, box):
    x1, y1, x2, y2 = box
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    depth_scale = max(bw, bh)

    points = {}
    for idx, lm in enumerate(pose):
        vis = float(getattr(lm, "visibility", 1.0))
        if vis < 0.18:
            continue
        gx = float(x1) + float(lm.x) * bw
        gy = float(y1) + float(lm.y) * bh
        gz = -float(lm.z) * depth_scale
        points[idx] = (gx, gy, gz)

    return points if len(points) >= 8 else None


def _extract_trajectory_point_3d(skeleton_points):
    if not skeleton_points:
        return None
    if 23 in skeleton_points and 24 in skeleton_points:
        l = np.asarray(skeleton_points[23], dtype=np.float32)
        r = np.asarray(skeleton_points[24], dtype=np.float32)
        c = (l + r) * 0.5
        return float(c[0]), float(c[1]), float(c[2])

    arr = np.asarray(list(skeleton_points.values()), dtype=np.float32)
    c = arr.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def _project_3d_point(point, width, height, scale):
    x, y, z = point
    px = int(width * 0.50 + x * scale + z * scale * 0.46)
    py = int(height * 0.84 - y * scale - z * scale * 0.24)
    return px, py


def _save_3d_skeleton_video(skeleton_frames, trajectory_points, output_path, fps=18.0):
    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    width = 960
    height = 720
    out_fps = max(float(fps), 1.0)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )
    if not writer.isOpened():
        return ""

    if not skeleton_frames:
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        for x in range(0, width, 40):
            color = (230, 230, 230) if x % 120 else (210, 210, 210)
            cv2.line(frame, (x, 0), (x, height), color, 1)
        for y in range(0, height, 40):
            color = (230, 230, 230) if y % 120 else (210, 210, 210)
            cv2.line(frame, (0, y), (width, y), color, 1)
        cv2.putText(
            frame,
            "3D skeleton trajectory: no pose",
            (120, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )
        for _ in range(int(max(24, out_fps * 2))):
            writer.write(frame)
        writer.release()
        _transcode_for_telegram(output_path, target_fps=out_fps)
        return output_path if os.path.exists(output_path) else ""

    max_frames = 540
    stride = max(1, int(math.ceil(len(skeleton_frames) / max_frames)))
    frames = skeleton_frames[::stride]
    traj = trajectory_points[::stride] if trajectory_points else []

    local_frames = []
    track_points = []
    body_spans = []
    for i, pts in enumerate(frames):
        arr = np.asarray(list(pts.values()), dtype=np.float32)
        if arr.size == 0:
            local_frames.append({})
            track_points.append(np.zeros(3, dtype=np.float32))
            body_spans.append(1.0)
            continue

        if 23 in pts and 24 in pts:
            center = (np.asarray(pts[23], dtype=np.float32) + np.asarray(pts[24], dtype=np.float32)) * 0.5
        else:
            center = arr.mean(axis=0)

        local = {}
        for idx, p in pts.items():
            local[idx] = np.asarray(p, dtype=np.float32) - center
        local_frames.append(local)

        local_arr = np.asarray(list(local.values()), dtype=np.float32)
        span_now = max(
            float(np.ptp(local_arr[:, 0])),
            float(np.ptp(local_arr[:, 1])),
            1e-3,
        )
        body_spans.append(span_now)

        if i < len(traj):
            track_points.append(np.asarray(traj[i], dtype=np.float32))
        else:
            track_points.append(center)

    body_norm = float(np.median(np.asarray(body_spans, dtype=np.float32)))
    if body_norm < 1e-3:
        body_norm = 1.0

    track_arr = np.asarray(track_points, dtype=np.float32)
    tr_center = track_arr.mean(axis=0)
    tr_span_x = max(float(np.ptp(track_arr[:, 0])), 1e-3)
    tr_span_y = max(float(np.ptp(track_arr[:, 1])), 1e-3)

    world_frames = []
    path_world = []
    for i, local in enumerate(local_frames):
        base_x = float((track_arr[i, 0] - tr_center[0]) / tr_span_x) * 3.0
        base_z = float((track_arr[i, 1] - tr_center[1]) / tr_span_y) * 1.8
        base_y = 0.0

        world = {}
        for idx, vv in local.items():
            lx = float(vv[0] / body_norm)
            ly = float(vv[1] / body_norm)
            lz = float(np.clip(vv[2] / body_norm, -1.2, 1.2))
            wx = base_x + lx * 0.95
            wy = base_y - ly * 2.4
            wz = base_z + lz * 0.40
            world[idx] = (wx, wy, wz)
        world_frames.append(world)
        path_world.append((base_x, 0.0, base_z))

    connections = list(pose_estimator.mp_pose.POSE_CONNECTIONS)

    total = len(world_frames)
    for i, pts in enumerate(world_frames):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        for gx in range(0, width, 40):
            color = (233, 233, 233) if gx % 120 else (210, 210, 210)
            cv2.line(frame, (gx, 0), (gx, height), color, 1)
        for gy in range(0, height, 40):
            color = (233, 233, 233) if gy % 120 else (210, 210, 210)
            cv2.line(frame, (0, gy), (width, gy), color, 1)

        if path_world:
            end = min(i + 1, len(path_world))
            prev = None
            for t_idx in range(end):
                px, py = _project_3d_point(path_world[t_idx], width, height, 140.0)
                if prev is not None:
                    c = int(80 + 175 * (t_idx / max(1, end - 1)))
                    col = (c // 3, 160, min(255, c + 30))
                    cv2.line(frame, prev, (px, py), col, 2)
                prev = (px, py)

        for a, b in connections:
            pa = pts.get(a)
            pb = pts.get(b)
            if pa is None or pb is None:
                continue
            x0, y0 = _project_3d_point(pa, width, height, 140.0)
            x1, y1 = _project_3d_point(pb, width, height, 140.0)
            cv2.line(frame, (x0, y0), (x1, y1), (70, 110, 220), 3)

        for _, p in pts.items():
            x, y = _project_3d_point(p, width, height, 140.0)
            cv2.circle(frame, (x, y), 4, (20, 170, 140), -1)

        cv2.putText(
            frame,
            "3D skeleton trajectory",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    _transcode_for_telegram(output_path, target_fps=out_fps)
    return output_path if os.path.exists(output_path) else ""


def process_frame(frame, model=None, trajectory_3d=None, skeleton_3d=None):
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
        model.update((x1, y1, x2, y2), pose)

        skeleton_points = _extract_skeleton_points_3d(pose, (x1, y1, x2, y2))
        if skeleton_points:
            if skeleton_3d is not None:
                skeleton_3d.append(skeleton_points)
            if trajectory_3d is not None:
                cp = _extract_trajectory_point_3d(skeleton_points)
                if cp is not None:
                    trajectory_3d.append(cp)

        action, action_prob = model.detect_action(pose)
        binary_scores = model.get_binary_scores()
        trajectory = model.get_recent_trajectory(max_points=90)

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
            (x1, max(25, y1 - 35)),
            28,
        )
        frame = draw_text(
            frame,
            (
                f"bin: статика {int(binary_scores['статика'] * 100)}% | "
                f"шаг {int(binary_scores['шаг'] * 100)}% | "
                f"присед {int(binary_scores['присед'] * 100)}%"
            ),
            (x1, min(h - 44, y2 + 23)),
            19,
        )
        frame = draw_text(
            frame,
            (
                f"bin: прыжок {int(binary_scores['прыжок'] * 100)}% | "
                f"мах рукой {int(binary_scores['мах рукой'] * 100)}%"
            ),
            (x1, min(h - 18, y2 + 45)),
            19,
        )

    return frame


def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    model = IntentPredictor()
    out = process_frame(img, model=model)
    cv2.imwrite(output_path, out)


def process_video(
    input_path,
    output_path,
    output_max_width=None,
    target_fps=None,
    trajectory_3d_video_path=None,
):
    default_3d_path = trajectory_3d_video_path or f"{os.path.splitext(output_path)[0]}_skeleton3d.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        shutil.copyfile(input_path, output_path)
        traj_video = _save_3d_skeleton_video([], [], default_3d_path, fps=18.0)
        return {
            "output_video_path": output_path,
            "trajectory_3d_video_path": traj_video,
            "trajectory_points": 0,
            "skeleton_frames": 0,
        }

    model = IntentPredictor()
    trajectory_points_3d = []
    skeleton_frames_3d = []

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
        (out_w, out_h),
    )
    if not out.isOpened():
        cap.release()
        shutil.copyfile(input_path, output_path)
        traj_video = _save_3d_skeleton_video([], [], default_3d_path, fps=out_fps)
        return {
            "output_video_path": output_path,
            "trajectory_3d_video_path": traj_video,
            "trajectory_points": 0,
            "skeleton_frames": 0,
        }

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
            processed = process_frame(
                frame,
                model=model,
                trajectory_3d=trajectory_points_3d,
                skeleton_3d=skeleton_frames_3d,
            )
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

    _transcode_for_telegram(output_path, target_fps=out_fps)
    traj_video = _save_3d_skeleton_video(
        skeleton_frames_3d,
        trajectory_points_3d,
        default_3d_path,
        fps=out_fps,
    )

    return {
        "output_video_path": output_path,
        "trajectory_3d_video_path": traj_video,
        "trajectory_points": len(trajectory_points_3d),
        "skeleton_frames": len(skeleton_frames_3d),
    }
