import cv2
import mediapipe as mp

resolucao_x = 1280
resolucao_y = 720
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

def find_hand_coordinates(frame, reverse_side=False):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    all_hands = []

    if result.multi_hand_landmarks:
        for hand_side, hand_marking in zip(result.multi_handedness, result.multi_hand_landmarks):
            info_hands = {}
            coords = []
            for marking in hand_marking.landmark:
                coord_x, coord_y, coord_z = int(marking.x * resolucao_x), int(marking.y * resolucao_y), int(marking.z * resolucao_x)
                coords.append((coord_x, coord_y, coord_z))

            info_hands['coords'] = coords
            if reverse_side:
                if hand_side.classification[0].label == 'Left':
                    info_hands['side'] = 'Right'
                else:
                    info_hands['side'] = 'Left'
            else:
                info_hands['side'] = hand_side.classification[0].label
            all_hands.append(info_hands)
            mp_draw.draw_landmarks(frame, hand_marking, mp_hands.HAND_CONNECTIONS)
    return frame, all_hands

def fingers_raised(detect_hands):
    fingers = []
    if detect_hands['side'] == 'Right':
        if detect_hands['coords'][4][0] < detect_hands['coords'][3][0]:
            fingers.append(True)
        else:
            fingers.append(False)
    else:
        if detect_hands['coords'][4][0] > detect_hands['coords'][3][0]:
            fingers.append(True)
        else:
            fingers.append(False)

    for finger_tip in [8, 12, 16, 20]:
        if detect_hands['coords'][finger_tip][1] < detect_hands['coords'][finger_tip-2][1]:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)

while True:
    has_frame, frame = cap.read()

    if not has_frame:
        print('Frame acabou!')
        break

    frame = cv2.flip(frame, 1)
    frame, all_hands = find_hand_coordinates(frame, reverse_side=True)
    if len(all_hands) == 1:
        info_hands_fingers = fingers_raised(all_hands[0])
        print(info_hands_fingers)

    cv2.imshow('Frames', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
