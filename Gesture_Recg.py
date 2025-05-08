import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def finger_states(hand_landmarks):
    fingers = []

    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
        fingers.append(1)
    else:
        fingers.append(0)

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)

    return fingers

def identify_gesture(fingers):
    total = sum(fingers)

    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"
    elif fingers == [0, 1, 0, 0, 0]:
        return "ONE"
    elif fingers == [0, 1, 1, 0, 0]:
        return "TWO"
    elif fingers == [0, 1, 1, 1, 0]:
        return "THREE"
    elif fingers == [0, 1, 1, 1, 1]:
        return "FOUR"
    elif fingers == [1, 1, 1, 1, 1]:
        return "FIVE"
    elif fingers == [1, 0, 0, 0, 0]:
        return "LIKE"
    elif fingers == [1, 1, 0, 0, 1]:
        return "ROCK"
    elif fingers == [0, 1, 1, 0, 0]:
        return "PEACE"
    elif fingers[0] == 1 and fingers[1] == 1:
        return "OK"
    else:
        return "UNKNOWN"

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = finger_states(hand_landmarks)
                gesture = identify_gesture(fingers)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign Language Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
