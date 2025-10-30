import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 각도 계산 함수
def calculate_angle(a, b, c):
    """
    세 점 a, b, c의 각도를 계산합니다.
    a, b, c는 (x, y) 형식의 튜플입니다.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # 벡터 계산
    ba = a - b
    bc = c - b
    
    # 코사인 유사도 계산
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# 쓰러짐 판정 함수
def is_fall_condition_met(keypoints):
    """
    mediapipe가 감지한 keypoints를 이용해 쓰러짐 여부를 판별합니다.
    """
    try:
        # Get keypoints for shoulders, hips, and knees
        left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # Ensure keypoints are within the frame (visibility > 0.5)
        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
                left_hip.visibility < 0.5 or right_hip.visibility < 0.5 or
                left_knee.visibility < 0.5 or right_knee.visibility < 0.5):
            return False

        # Calculate the midpoint of shoulders and hips
        shoulder = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
        hip = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
        knee = [(left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2]

        # Calculate the angle between shoulder-hip and hip-knee lines
        angle_shoulder_hip_knee = calculate_angle(shoulder, hip, knee)

        # If both angles are below or above certain thresholds, consider it as falling
        if (angle_shoulder_hip_knee < 30 or angle_shoulder_hip_knee > 150) :
            return False
        else:
            return True
    except:
        return False

cap = cv2.VideoCapture('fall_final.mp4')

# 포즈 객체 초기화
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # 쓰러짐 상태 추적 변수
    fall_detected = False
    fall_start_time = None
    required_duration = 0.2  # 초 단위

    # 플래싱 효과를 위한 변수
    flashing = False
    last_flash_time = 0
    flash_interval = 0.5  # 초 단위
    mask_visible = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오를 불러올 수 없습니다.")
            break

        frame = cv2.resize(frame, (1000, 600))

        # 이미지 좌우 반전
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # 초기화
        angles = {}

        if results.pose_landmarks:
            # Mediapipe 랜드마크 추출
            keypoints = results.pose_landmarks.landmark

            # 각도 판정
            fall_condition = is_fall_condition_met(keypoints)

            current_time = time.time()

            if fall_condition:
                if not fall_detected:
                    fall_start_time = current_time
                    fall_detected = True
                    flashing = False
                else:
                    elapsed_time = current_time - fall_start_time
                    if elapsed_time >= required_duration:
                        # 쓰러짐 감지 알람 텍스트 표시
                        cv2.putText(image, 'Fall Detected', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # 플래싱 효과 시작
                        flashing = True
            else:
                fall_detected = False
                fall_start_time = None
                # 플래싱 효과 중지
                flashing = False
                mask_visible = False

            # 플래싱 효과 구현
            if flashing:
                if current_time - last_flash_time >= flash_interval:
                    mask_visible = not mask_visible
                    last_flash_time = current_time
                if mask_visible:
                    # 붉은색 투명 마스크 생성
                    overlay = image.copy()
                    red_color = (0, 0, 255)  # BGR 형식
                    alpha = 0.4  # 투명도 (0.0 ~ 1.0)
                    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), red_color, -1)
                    # 원본 이미지와 오버레이 이미지 합성
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 화면에 이미지 표시
        cv2.imshow('Fall Detection', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()