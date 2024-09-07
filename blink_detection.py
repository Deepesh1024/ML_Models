import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def blink_detection():
    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 1
    COUNTER = 0
    TOTAL = 0
    blink_timestamps = []

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                h, w, _ = frame.shape

                leftEye = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in LEFT_EYE_INDICES])
                rightEye = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in RIGHT_EYE_INDICES])

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # cv2.polylines(frame, [leftEye.astype(np.int32)], True, (0, 255, 0), 1)
                # cv2.polylines(frame, [rightEye.astype(np.int32)], True, (0, 255, 0), 1)
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        blink_time = time.time() - start_time
                        blink_timestamps.append(blink_time)
                        COUNTER = 0

                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return TOTAL, blink_timestamps
total_blinks, blink_timestamps = blink_detection()
print("Blinks occurred at the following time frames (in seconds since start):")
for timestamp in blink_timestamps:
     print(f"{timestamp:.2f} seconds")
print(f'Total blinks {len(blink_timestamps)}')
