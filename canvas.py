import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import copy

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Color settings
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
color_index = 0  # Start with blue

# Deques for drawing points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes for tracking color points
blue_index = green_index = red_index = yellow_index = 0

# History stacks for undo and redo
history = []
redo_stack = []

# Initialize the canvas
paintWindow = np.ones((471, 636, 3), dtype="uint8") * 255
button_width = 50
button_height = 30

paintWindow = cv2.rectangle(paintWindow, (40, 1), (40 + button_width, 1 + button_height), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (100, 1), (100 + button_width, 1 + button_height), colors[0], 2)  # Blue
paintWindow = cv2.rectangle(paintWindow, (160, 1), (160 + button_width, 1 + button_height), colors[1], 2)  # Green
paintWindow = cv2.rectangle(paintWindow, (220, 1), (220 + button_width, 1 + button_height), colors[2], 2)  # Red
paintWindow = cv2.rectangle(paintWindow, (280, 1), (280 + button_width, 1 + button_height), colors[3], 2)  # Yellow
paintWindow = cv2.rectangle(paintWindow, (340, 1), (340 + button_width, 1 + button_height), (128, 128, 128), 2)  # Undo
paintWindow = cv2.rectangle(paintWindow, (400, 1), (400 + button_width, 1 + button_height), (128, 128, 128), 2)  # Redo

cv2.putText(paintWindow, "CLEAR", (45, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (105, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (165, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (225, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (285, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "UNDO", (345, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWindow, "REDO", (405, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(frame_rgb)

    # Draw rectangles and text on the frame
    frame = cv2.rectangle(frame, (40, 1), (40 + button_width, 1 + button_height), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (100, 1), (100 + button_width, 1 + button_height), colors[0], 2)
    frame = cv2.rectangle(frame, (160, 1), (160 + button_width, 1 + button_height), colors[1], 2)
    frame = cv2.rectangle(frame, (220, 1), (220 + button_width, 1 + button_height), colors[2], 2)
    frame = cv2.rectangle(frame, (280, 1), (280 + button_width, 1 + button_height), colors[3], 2)
    frame = cv2.rectangle(frame, (340, 1), (340 + button_width, 1 + button_height), (128, 128, 128), 2)
    frame = cv2.rectangle(frame, (400, 1), (400 + button_width, 1 + button_height), (128, 128, 128), 2)
    cv2.putText(frame, "CLEAR", (45, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (105, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (165, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "RED", (225, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (285, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "UNDO", (345, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "REDO", (405, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * frame.shape[1])
                lmy = int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if landmarks:
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])

            if (thumb[1] - center[1]) < 30:
                # Start a new drawing
                bpoints.append(deque(maxlen=1024))
                gpoints.append(deque(maxlen=1024))
                rpoints.append(deque(maxlen=1024))
                ypoints.append(deque(maxlen=1024))

                blue_index += 1
                green_index += 1
                red_index += 1
                yellow_index += 1

            elif center[1] <= button_height:
                if 40 <= center[0] <= 40 + button_width:  # Clear Button
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    ypoints = [deque(maxlen=1024)]
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[button_height + 1:, :, :] = 255
                    history.clear()
                    redo_stack.clear()
                elif 100 <= center[0] <= 100 + button_width:
                    color_index = 0  # Blue
                elif 160 <= center[0] <= 160 + button_width:
                    color_index = 1  # Green
                elif 220 <= center[0] <= 220 + button_width:
                    color_index = 2  # Red
                elif 280 <= center[0] <= 280 + button_width:
                    color_index = 3  # Yellow
                elif 340 <= center[0] <= 340 + button_width:  # Undo Button
                    if history:
                        redo_stack.append(copy.deepcopy(history.pop()))
                        if history:
                            bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index = copy.deepcopy(history[-1])
                        else:
                            bpoints = [deque(maxlen=1024)]
                            gpoints = [deque(maxlen=1024)]
                            rpoints = [deque(maxlen=1024)]
                            ypoints = [deque(maxlen=1024)]
                            blue_index = green_index = red_index = yellow_index = 0
                            paintWindow[button_height + 1:, :, :] = 255
                elif 400 <= center[0] <= 400 + button_width:  # Redo Button
                    if redo_stack:
                        history.append(copy.deepcopy(redo_stack.pop()))
                        bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index = copy.deepcopy(history[-1])

            else:
                if color_index == 0:
                    bpoints[blue_index].appendleft(center)
                elif color_index == 1:
                    gpoints[green_index].appendleft(center)
                elif color_index == 2:
                    rpoints[red_index].appendleft(center)
                elif color_index == 3:
                    ypoints[yellow_index].appendleft(center)

                # Save the current state to history for undo
                history.append(copy.deepcopy((bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index)))
                redo_stack.clear()  # Clear the redo stack when a new action is made

    else:
        bpoints.append(deque(maxlen=1024))
        gpoints.append(deque(maxlen=1024))
        rpoints.append(deque(maxlen=1024))
        ypoints.append(deque(maxlen=1024))
        blue_index += 1
        green_index += 1
        red_index += 1
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
