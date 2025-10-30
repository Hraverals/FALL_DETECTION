# ONLY mediapipe (add second counter maintaning feature)
# 현재 최종본

import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    세 점 a, b, c가 이루는 각도(도)를 계산 (b가 꼭지점)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def is_fall_condition_met(keypoints):
    """
    몸통(어깨-엉덩이) 기울기 기반으로 넘어짐(누움) 상태 판별.

    - 다리 들어올림 동작에서도 몸통은 수직에 가까우므로 오탐을 줄임
    - 몸통이 수평에 가까울수록(수직과의 각도 ↑) 넘어짐으로 간주
    """
    try:
        left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # 가시성 체크
        if (
            left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
            left_hip.visibility < 0.5 or right_hip.visibility < 0.5
        ):
            return False

        # 어깨/엉덩이 중앙점
        shoulder = np.array([
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
        ])
        hip = np.array([
            (left_hip.x + right_hip.x) / 2.0,
            (left_hip.y + right_hip.y) / 2.0,
        ])

        torso_vec = hip - shoulder
        torso_len = np.linalg.norm(torso_vec)
        if torso_len < 1e-6:
            return False

        # 수직 벡터 (영상 좌표계에서 아래쪽이 +y)
        vertical = np.array([0.0, 1.0])
        cos_tilt = np.dot(torso_vec, vertical) / (torso_len * np.linalg.norm(vertical))
        cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
        tilt_deg = np.degrees(np.arccos(cos_tilt))

        # 55도 이상이면 몸통이 수평에 가까움 -> 넘어짐으로 판정
        return tilt_deg > 55.0
    except Exception:
        return False


cap = cv2.VideoCapture('fall_final.mp4')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # 지속 상태 추적 변수
    fall_candidate_since = None   # 넘어짐 조건이 연속 유지되기 시작한 시각
    fall_state = False            # 넘어짐 상태 확정 여부
    fall_state_since = None       # 넘어짐 상태 확정 시각
    required_duration = 0.2       # 짧은 디바운스(초)
    # 깜빡임(플래싱) 효과 변수
    flashing = False
    mask_visible = False
    last_flash_time = 0.0
    flash_interval = 0.5          # 초 단위 간격

    # tracking lost 후 돌아올 때 복원 위한 상태 변수
    RESUME_GRACE_SEC = 5.0
    paused_elapsed = 0.0
    last_tracking_lost_time = None
    was_in_fall_when_lost = False
    # classification flip 이후 복원 위한 상태 변수
    last_fall_ended_time = None
    was_in_fall_when_ended = False
    paused_elapsed_end = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오를 불러오지 못했습니다.")
            break

        frame = cv2.resize(frame, (1000, 600))

        # 좌우 반전 (캠 쓸때 거울 효과용) 및 RGB 변환
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        current_time = time.time()

        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark

            fall_condition = is_fall_condition_met(keypoints)
            current_time = time.time()

            # 디바운스 및 상태 확정/유지
            # 낙상 중 추적이 끊기더라도 일정 허용 범위 (현재 5초) 내에서 추적 복원 -> 낙상 시간 유지
            if (not fall_state) and was_in_fall_when_lost and last_tracking_lost_time is not None and (current_time - last_tracking_lost_time) <= RESUME_GRACE_SEC:
                if fall_condition:
                    fall_state = True
                    fall_state_since = current_time - paused_elapsed
                    was_in_fall_when_lost = False
                    last_tracking_lost_time = None
                    paused_elapsed = 0.0
                else:
                    if (current_time - last_tracking_lost_time) > RESUME_GRACE_SEC:
                        was_in_fall_when_lost = False
                        last_tracking_lost_time = None
                        paused_elapsed = 0.0

            # normal state update
            if fall_state:
                if not fall_condition:
                    # classification-based exit 에서 마지막 상태 기록
                    if fall_state_since is not None:
                        paused_elapsed_end = max(0.0, current_time - fall_state_since)
                    else:
                        paused_elapsed_end = 0.0
                    last_fall_ended_time = current_time
                    was_in_fall_when_ended = True
                    fall_candidate_since = None
                    fall_state = False
                    fall_state_since = None
            else:
                if fall_condition:
                    if fall_candidate_since is None:
                        fall_candidate_since = current_time
                    if (current_time - fall_candidate_since) >= required_duration:
                        fall_state = True
                        fall_state_since = fall_candidate_since
                else:
                    fall_candidate_since = None

            # 낙상 중 잠시 자세가 바뀌거나 분류가 뒤집혀도 (classification flip) 일정 허용 범위 (현재 5초) 내에서 낙상 시간 유지
            if (not fall_state) and was_in_fall_when_ended and last_fall_ended_time is not None and (current_time - last_fall_ended_time) <= RESUME_GRACE_SEC:
                if fall_condition:
                    fall_state = True
                    fall_state_since = current_time - paused_elapsed_end
                    was_in_fall_when_ended = False
                    last_fall_ended_time = None
                    paused_elapsed_end = 0.0
                else:
                    if (current_time - last_fall_ended_time) > RESUME_GRACE_SEC:
                        was_in_fall_when_ended = False
                        last_fall_ended_time = None
                        paused_elapsed_end = 0.0

            # 상태 텍스트 + 플래싱 + 경과시간 표시 (지속 모니터링)
            if fall_state and fall_state_since is not None:
                # 플래싱 토글 업데이트
                flashing = True
                if current_time - last_flash_time >= flash_interval:
                    mask_visible = not mask_visible
                    last_flash_time = current_time

                # 마스크 적용 (반투명 붉은색)
                if mask_visible:
                    overlay = image.copy()
                    red_color = (0, 0, 255)  # BGR
                    alpha = 0.35
                    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), red_color, -1)
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                # 경과 시간(초) 계산 및 표시
                elapsed = max(0.0, current_time - fall_state_since)
                seconds_str = f"{elapsed:0.1f}s"

                # 상태 텍스트
                if elapsed >= 30.0:
                    status_text = 'EMERGENCY'
                else:
                    status_text = 'FALL DETECTION'

                cv2.putText(
                    image, status_text, (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
                )
                cv2.putText(
                    image, seconds_str, (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
                )
            else:
                # 상태 해제 시 플래싱도 리셋
                flashing = False
                mask_visible = False

            # 랜드마크 시각화
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 화면 출력
        if not results.pose_landmarks:
            # tracking lost
            if fall_state and fall_state_since is not None:
                paused_elapsed = max(0.0, current_time - fall_state_since)
                last_tracking_lost_time = current_time
                was_in_fall_when_lost = True
                fall_candidate_since = None
                fall_state = False
                fall_state_since = None
            else:
                if last_tracking_lost_time is not None and (current_time - last_tracking_lost_time) > RESUME_GRACE_SEC:
                    was_in_fall_when_lost = False
                    last_tracking_lost_time = None
                    paused_elapsed = 0.0
            # 추적 끊길 시에 분류 복원 관련 변수 (flags) 초기화
            was_in_fall_when_ended = False
            last_fall_ended_time = None
            paused_elapsed_end = 0.0
            flashing = False
            mask_visible = False

        cv2.imshow('Fall Detection', image)

        # 종료 키
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()