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
                    info_hands['lado'] = 'Right'
                else:
                    info_hands['lado'] = 'Left'
            else:
                info_hands['lado'] = hand_side.classification[0].label
            all_hands.append(info_hands)
            mp_draw.draw_landmarks(frame, hand_marking, mp_hands.HAND_CONNECTIONS)

    return frame, all_hands


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

    cv2.imshow('Frames', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
