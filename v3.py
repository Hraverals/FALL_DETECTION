import cv2
import cvzone
import time
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# ----------------------------
# YOLO 모델 로드
# ----------------------------
model = YOLO('yolov8s.pt')
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# ----------------------------
# MediaPipe Pose 설정
# ----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# MediaPipe 낙상 판정 함수
def is_fall_condition_met(keypoints):
    try:
        left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
            left_hip.visibility < 0.5 or right_hip.visibility < 0.5 or
            left_knee.visibility < 0.5 or right_knee.visibility < 0.5):
            return False

        shoulder = [(left_shoulder.x + right_shoulder.x)/2, (left_shoulder.y + right_shoulder.y)/2]
        hip = [(left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2]
        knee = [(left_knee.x + right_knee.x)/2, (left_knee.y + right_knee.y)/2]

        angle = calculate_angle(shoulder, hip, knee)
        if 30 < angle < 150:
            return True
        else:
            return False
    except:
        return False

# ----------------------------
# 웹캠 연결
# ----------------------------
cap = cv2.VideoCapture(0)
fall_detected = False
fall_start_time = None
required_duration = 0.2
flashing = False
last_flash_time = 0
flash_interval = 0.5
mask_visible = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1000, 600))
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ----------------------------
        # YOLO Detection
        # ----------------------------
        results = model(image, stream=True)
        fall_box = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_idx = int(box.cls[0])
                class_name = classnames[cls_idx]
                confidence = box.conf[0]

                if class_name != 'person' or confidence < 0.8:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                threshold = h - w

                cvzone.cornerRect(image, [x1, y1, w, h], l=30, rt=6)
                cvzone.putTextRect(image, f'{class_name} {int(confidence*100)}%', [x1 + 8, y1 - 12], thickness=2, scale=2)

                # Bounding Box 기준 낙상
                if threshold < 0:
                    fall_box = True

                # ----------------------------
                # MediaPipe Pose Detection
                # ----------------------------
                person_roi = image[y1:y2, x1:x2]
                person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(person_rgb)

                fall_pose = False
                if pose_results.pose_landmarks:
                    keypoints = pose_results.pose_landmarks.landmark
                    fall_pose = is_fall_condition_met(keypoints)
                    mp_drawing.draw_landmarks(person_roi, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # ----------------------------
                # 최종 낙상 판단 (두 기준 모두 만족)
                # ----------------------------
                if fall_box and fall_pose:
                    fall_detected = True
                    cvzone.putTextRect(image, "FALL DETECTED", [x1, y1 - 50], scale=2, thickness=2, colorR=(0,0,255), colorT=(255,255,255))

        # ----------------------------
        # 화면 표시 및 플래싱 효과
        # ----------------------------
        current_time = time.time()
        if fall_detected:
            if not flashing:
                flashing = True
                fall_start_time = current_time
            elif current_time - fall_start_time >= required_duration:
                if current_time - last_flash_time >= flash_interval:
                    mask_visible = not mask_visible
                    last_flash_time = current_time
                if mask_visible:
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0,0), (image.shape[1], image.shape[0]), (0,0,255), -1)
                    image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        else:
            fall_detected = False
            flashing = False
            mask_visible = False

        cv2.imshow("Fall Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
