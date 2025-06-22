import cv2
import mediapipe as mp
import numpy as np
import math

# 初始化
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 背景顏色
BG_COLOR = (1000, 1000, 1000)  # gray
ExStatus = False
countEx = 0

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5)

def FindAngleF(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    if ang < 0:
        ang += 360
    if ang >= 360 - ang:
        ang = 360 - ang
    return ang

def countExF(HandAngle):
    global ExStatus, countEx
    if HandAngle < 100 and not ExStatus:
        countEx += 1
        ExStatus = True
    elif HandAngle > 100:
        ExStatus = False
    return countEx

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgH, imgW = image.shape[0], image.shape[1]
        results = pose.process(image)

        if results.pose_landmarks:
            a = np.array([results.pose_landmarks.landmark[12].x * imgW, results.pose_landmarks.landmark[12].y * imgH])
            b = np.array([results.pose_landmarks.landmark[14].x * imgW, results.pose_landmarks.landmark[14].y * imgH])
            c = np.array([results.pose_landmarks.landmark[16].x * imgW, results.pose_landmarks.landmark[16].y * imgH])

            d = np.array([results.pose_landmarks.landmark[11].x * imgW, results.pose_landmarks.landmark[11].y * imgH])
            e = np.array([results.pose_landmarks.landmark[13].x * imgW, results.pose_landmarks.landmark[13].y * imgH])
            f = np.array([results.pose_landmarks.landmark[15].x * imgW, results.pose_landmarks.landmark[15].y * imgH])

            # 計算角度
            RightHandAngle = FindAngleF(a, b, c)
            LeftHandAngle = FindAngleF(d, e, f)

            # 更新次數
            countExF(RightHandAngle)
            countExF(LeftHandAngle)

            # 畫出點位
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # 顯示右手角度
            x13, y13 = round((1 - results.pose_landmarks.landmark[13].x) * imgW), int(results.pose_landmarks.landmark[13].y * imgH)
            if 0 < x13 < imgW and 0 < y13 < imgH:
                cv2.putText(image, str(round(RightHandAngle, 2)), (x13, y13), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

            # 顯示左手角度
            x14, y14 = round((1 - results.pose_landmarks.landmark[14].x) * imgW), int(results.pose_landmarks.landmark[14].y * imgH)
            if 0 < x14 < imgW and 0 < y14 < imgH:
                cv2.putText(image, str(round(LeftHandAngle, 2)), (x14, y14), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

            # 顯示計數
            cv2.putText(image, str(countEx), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()