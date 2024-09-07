import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def draw_keypoints(frame, keypoints, confidence_threshold, indices):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    coordinates = {}
    for ind in indices:
        ky, kx, kp_conf = shaped[ind]
        if kp_conf > confidence_threshold:
            coordinates[ind] = (int(kx) , int(ky))
            cv2.circle(frame, (int(kx), int(ky)), 4, (250 , 212, 22, ), -1)
    return coordinates

EDGES = {
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c'
}
try:
    from tensorflow.lite.experimental import Interpreter
    from tensorflow.lite.experimental.acceleration import get_input_details, get_output_details
    print("Running on GPU")
    delegate = tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')
    interpreter = tf.lite.Interpreter(model_path=r'3.tflite', experimental_delegates=[delegate])
except Exception as e:
    print("GPU not available, running on CPU")
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


interpreter = tf.lite.Interpreter(model_path=r'3.tflite')
interpreter.allocate_tensors()
def draw_hand_landmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
            for idx, landmark in enumerate(landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    img = frame.copy()
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks.
    result = hands.process(rgb_frame)

    # Draw hand landmarks on the frame.
    draw_hand_landmarks(frame, result.multi_hand_landmarks)
    cv2.imshow('Hand Pose', frame)
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    indices = [5, 6, 7, 8, 9, 10]
    draw_connections(frame, keypoints_with_scores, EDGES, 0.3)
    coordinates = draw_keypoints(frame, keypoints_with_scores, 0.3, indices)
    cv2.imshow('MoveNet Lightning', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()