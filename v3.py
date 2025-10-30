# YOLO + mediapipe
# v3.py가 v4.py보다 느린 핵심 이유는 YOLO 때문

import cv2
import cvzone
import time
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import math

try:
    import torch
except Exception:
    torch = None

# Debug overlay: set True to draw live metrics (angles, distances)
DEBUG = False

# ----------------------------
# Model setup (device/precision)
# ----------------------------
model = YOLO('yolov8s.pt')

# Prefer GPU if available. Use FP16 on CUDA for speed; fuse layers on CPU.
if torch is not None:
    if torch.cuda.is_available():
        model.to('cuda')
        try:
            model.model.half()  # fp16 inference
        except Exception:
            pass
    else:
        try:
            model.fuse()  # fuse Conv+BN for small CPU speedup
        except Exception:
            pass

# Optional classnames for nicer labels if present
classnames = []
try:
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()
except Exception:
    classnames = []

# ----------------------------
# MediaPipe Pose setup
# ----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Angle helper (not used in current logic, kept for reference)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Decide fall condition from pose landmarks
# Heuristic: horizontal torso + ankle/torso alignment (or compact vertical extent when ankles not visible)
def is_fall_condition_met(keypoints):
    try:
        l_sh = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hp = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hp = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_an = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_an = keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Basic visibility requirement for torso points
        if (l_sh.visibility < 0.4 or r_sh.visibility < 0.4 or
                l_hp.visibility < 0.4 or r_hp.visibility < 0.4):
            return False

        shoulder = [(l_sh.x + r_sh.x) / 2.0, (l_sh.y + r_sh.y) / 2.0]
        hip = [(l_hp.x + r_hp.x) / 2.0, (l_hp.y + r_hp.y) / 2.0]

        # Torso angle relative to vertical axis (0 = vertical, 90 = horizontal)
        vx = shoulder[0] - hip[0]
        vy = shoulder[1] - hip[1]
        vertical_angle_deg = float(np.degrees(np.arctan2(abs(vx), abs(vy) + 1e-6)))

        # Consider as candidate when sufficiently close to horizontal
        HORIZONTAL_THR_DEG = 40
        if vertical_angle_deg < HORIZONTAL_THR_DEG:
            # Closer to vertical → not a fall
            return False

        # If (at least one) ankles are visible, check ankle–torso vertical distance
        ankles = []
        if l_an.visibility >= 0.4:
            ankles.append(l_an.y)
        if r_an.visibility >= 0.4:
            ankles.append(r_an.y)
        if len(ankles) > 0:
            ankle_y_mean = float(np.mean(ankles))
            torso_y_mean = (shoulder[1] + hip[1]) / 2.0
            # Smaller difference indicates lying posture
            if abs(ankle_y_mean - torso_y_mean) <= 0.33:
                return True
            else:
                return False

        # If ankles are not visible, use compact vertical extent of torso as fallback
        key_y = [shoulder[1], hip[1]]
        vertical_extent = max(key_y) - min(key_y)
        if vertical_extent <= 0.40:
            return True
        # Fallback keeps original behavior (conservative): treat as fall
        return True
    except Exception:
        return False

# ----------------------------
# Video source (0 = default webcam)
# ----------------------------
cap = cv2.VideoCapture(0)

# UI state flags
fall_detected = False
fall_start_time = None
required_duration = 0.2  # delay before starting flashing overlay
flashing = False
last_flash_time = 0
flash_interval = 0.5
mask_visible = False

with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:
    frame_count = 0
    YOLO_EVERY_N = 1  # run YOLO every N frames
    last_boxes = []

    # Hysteresis state (sensitivity tuning)
    fall_state = False
    fall_streak = 0
    stand_streak = 0
    # More sensitive hysteresis and emergency timer
    FALL_ON_FRAMES = 1
    FALL_OFF_FRAMES = 6
    EMERGENCY_AFTER = 30.0
    emergency_state = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize while keeping aspect ratio (fix height to 600)
        h0, w0 = frame.shape[:2]
        target_h = 600
        scale = target_h / float(h0)
        target_w = int(w0 * scale)
        image = cv2.resize(frame, (target_w, target_h))

        # Optional horizontal flip for display symmetry
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ----------------------------
        # YOLO detection (person only, throttled)
        # ----------------------------
        fall_candidate = False

        run_yolo = (frame_count % YOLO_EVERY_N == 0)
        if run_yolo:
            r = model.predict(
                image,
                classes=[0],      # person only
                conf=0.55,        # slightly lower for sensitivity
                verbose=False,
                stream=False
            )[0]
            last_boxes = r.boxes

        boxes = last_boxes
        for box in boxes:
            cls_idx = int(box.cls[0]) if hasattr(box, 'cls') else 0
            confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            if cls_idx != 0 or confidence < 0.55:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Clip to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(image.shape[1] - 1, x2); y2 = min(image.shape[0] - 1, y2)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue

            # Draw box and label (disable for extra speed if needed)
            cvzone.cornerRect(image, [x1, y1, w, h], l=30, rt=6)
            label = f"person {int(confidence*100)}%"
            if classnames and 0 < len(classnames):
                try:
                    label = f"{classnames[cls_idx]} {int(confidence*100)}%"
                except Exception:
                    pass
            cvzone.putTextRect(image, label, [x1 + 8, y1 - 12], thickness=2, scale=2)

            # ----------------------------
            # MediaPipe Pose on ROI
            # ----------------------------
            person_roi = image[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(person_rgb)

            fall_pose = False
            dbg_text = None
            if pose_results.pose_landmarks:
                keypoints = pose_results.pose_landmarks.landmark
                fall_pose = is_fall_condition_met(keypoints)
                if DEBUG:
                    try:
                        l_sh = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        r_sh = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        l_hp = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
                        r_hp = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
                        l_an = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                        r_an = keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                        shoulder = [(l_sh.x + r_sh.x)/2.0, (l_sh.y + r_sh.y)/2.0]
                        hip = [(l_hp.x + r_hp.x)/2.0, (l_hp.y + r_hp.y)/2.0]
                        vx = shoulder[0] - hip[0]
                        vy = shoulder[1] - hip[1]
                        vertical_angle_deg = float(np.degrees(np.arctan2(abs(vx), abs(vy)+1e-6)))
                        ankles = []
                        if l_an.visibility >= 0.4:
                            ankles.append(l_an.y)
                        if r_an.visibility >= 0.4:
                            ankles.append(r_an.y)
                        ankle_y_mean = float(np.mean(ankles)) if len(ankles)>0 else float('nan')
                        torso_y_mean = (shoulder[1] + hip[1]) / 2.0
                        ankle_delta = abs(ankle_y_mean - torso_y_mean) if not math.isnan(ankle_y_mean) else float('nan')
                        vertical_extent = abs(max([shoulder[1], hip[1]]) - min([shoulder[1], hip[1]]))
                        dbg_text = f"angV={vertical_angle_deg:.1f} ankD={ankle_delta if not math.isnan(ankle_delta) else -1:.2f} ext={vertical_extent:.2f}"
                    except Exception:
                        pass
                mp_drawing.draw_landmarks(person_roi, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Aggregate per-person result
            if fall_pose:
                fall_candidate = True
                if DEBUG and dbg_text:
                    cv2.putText(image, dbg_text, (x1, max(0, y1-60)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2, cv2.LINE_AA)
            elif DEBUG and dbg_text:
                cv2.putText(image, dbg_text, (x1, max(0, y1-60)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        # ----------------------------
        # Frame-level state update and alerts
        # ----------------------------
        current_time = time.time()

        # Hysteresis streaks
        if fall_candidate:
            fall_streak += 1
            stand_streak = 0
        else:
            stand_streak += 1
            fall_streak = 0

        # Enter fall state
        if not fall_state and fall_streak >= FALL_ON_FRAMES:
            fall_state = True
            fall_start_time = current_time
            flashing = False
            emergency_state = False

        # Exit fall state
        if fall_state and stand_streak >= FALL_OFF_FRAMES:
            fall_state = False
            flashing = False
            mask_visible = False
            emergency_state = False

        # Visual alerts when in fall state
        if fall_state:
            # Start flashing after a short delay
            if not flashing:
                if current_time - fall_start_time >= required_duration:
                    flashing = True
            if flashing and (current_time - last_flash_time >= flash_interval):
                mask_visible = not mask_visible
                last_flash_time = current_time
            if flashing and mask_visible:
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), -1)
                image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

        # Additional status text: EMERGENCY after 30s, otherwise pre-flash FALL DETECTED
        if fall_state:
            if (current_time - fall_start_time) >= EMERGENCY_AFTER:
                emergency_state = True
            if emergency_state:
                cv2.putText(image, 'EMERGENCY', (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(image, 'FALL DETECTED', (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Fall Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

cap.release()
cv2.destroyAllWindows()

