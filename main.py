#gesture-recognition-opencv-mediapipe
import cv2
import mediapipe as mp
import numpy as np

# === Initialize MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# === Webcam Capture ===
cap = cv2.VideoCapture(0)

def get_gesture(landmarks):
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    def dist(a, b):
        return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))

    # Distances to wrist
    d_thumb = dist(wrist, thumb_tip)
    d_index = dist(wrist, index_tip)
    d_middle = dist(wrist, middle_tip)
    d_ring = dist(wrist, ring_tip)
    d_pinky = dist(wrist, pinky_tip)

    # === Gestures ===

    # Fist
    if all(dist(wrist, landmarks[i]) < 0.1 for i in [4, 8, 12, 16, 20]):
        return "Fist"

    # Open Palm
    elif all(dist(wrist, landmarks[i]) > 0.2 for i in [8, 12, 16, 20]):
        return "Open Palm"

    # Thumbs Up
    elif d_thumb > 0.25 and all(dist(wrist, landmarks[i]) < 0.15 for i in [8, 12, 16, 20]):
        return "Thumbs Up"

    # Peace Sign (index and middle fingers up)
    elif d_index > 0.2 and d_middle > 0.2 and d_ring < 0.1 and d_pinky < 0.1:
        return "Peace"

    # OK Sign (thumb and index tip close together)
    elif dist(thumb_tip, index_tip) < 0.05 and d_middle > 0.15:
        return "OK"

    # Call Me (thumb and pinky extended)
    elif d_thumb > 0.2 and d_pinky > 0.2 and d_index < 0.1 and d_middle < 0.1 and d_ring < 0.1:
        return "Call Me"

    else:
        return "Unknown"


# === Main Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "No Hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get gesture from landmarks
            gesture = get_gesture(hand_landmarks.landmark)

    # Display result
    cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition (No ML)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
