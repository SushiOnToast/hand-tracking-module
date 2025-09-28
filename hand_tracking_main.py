"""
The base code for getting hand detection to work
"""

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

p_time = 0  # previous time
c_time = 0  # current time#

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb) # fiding the hands from the image (Must be rgb)

    # print(results.multi_hand_landmarks)

    # finding the hand landmarks and drawing
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark): # id, landmark
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # center x and center y ( as a pixel value )
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(
                img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
