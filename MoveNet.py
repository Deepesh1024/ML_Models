import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

#DrawKeypoints

def draw_keypoints(frame, keypoints, confidence_threshold):
    y,x,z = frame.shape
    shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))

    for kp in shaped:
        ky,kx,kp_conf = kp
        if kp_conf > confidence_threshold :
            cv2.circle(frame, (int(kx),int(ky)),4,(0,255,0),-1)

EDGES = {(5, 7): 'm',(7, 9): 'm',(6, 8): 'c',(8, 10): 'c',(5, 6): 'y'}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))

    for edge, color in edges.items():
        p1,p2 = edge
        y1,x1,c1 = shaped[p1]
        y2,x2,c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),4)


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    # setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_scores)

    # Rendering
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    cv2.imshow('MoveNet Lightning', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

right_shoulder = keypoints_with_scores[0][0][6]
right_elbow = keypoints_with_scores[0][0][8]
right_wrist = keypoints_with_scores[0][0][10]
print("Right Shoulder = ",right_shoulder)
print("Right Elbow = ",right_elbow)
print("Right Wrist = ", right_wrist)
orientation = [right_shoulder, right_elbow, right_wrist]
print(orientation)


# keyp = [(5, 7),(7, 9),(6, 8),(8, 10),(5, 6)]
# def calculate_vector(point1, point2):
#     return np.array(point2) - np.array(point1)
#
# def calculate_angle(vector1, vector2):
#     unit_vector1 = vector1 / np.linalg.norm(vector1)
#     unit_vector2 = vector2 / np.linalg.norm(vector2)
#     dot_product = np.dot(unit_vector1, unit_vector2)
#     angle = np.arccos(dot_product)
#     # print(np.degrees(angle))
#     return np.degrees(angle)  # Convert to degrees if needed
#
#
# pointA1 = keyp[0]  # First point of edge 1
# pointA2 = keyp[6]  # Second point of edge 1
# pointB1 = keyp[8]  # First point of edge 2
# pointB2 = keyp[10]  # Second point of edge 2
#
# # Calculate the vectors
# vector1 = calculate_vector(pointA1, pointA2)
# vector2 = calculate_vector(pointB1, pointB2)
#
# # Calculate the angle
# angle = calculate_angle(vector1, vector2)
# print(np.degrees(angle))
# print(f"The angle between the edges is {angle:.2f} degrees")
#
# print(np.degrees(angle))
