"""
Hand tracking module with interactive ui, bounding boxes, left and right classification, and more
"""

import cv2
import mediapipe as mp
import time
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_num_hands,
                                         min_detection_confidence=self.min_detection_confidence,
                                         min_tracking_confidence=self.min_tracking_confidence)

        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw_landmarks=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)  # finding hands

        if self.results.multi_hand_landmarks and draw_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmark, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, draw_labels=True, draw_bounding_boxes=True):
        all_hands = []

        if self.results.multi_hand_landmarks:
            for hand_id, hand_landmark in enumerate(self.results.multi_hand_landmarks):
                lm_list = []
                x_list, y_list = [], []
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    x_list.append(cx)
                    y_list.append(cy)

                    if draw_labels:
                        cv2.putText(img, str(id), (cx, cy),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                if draw_bounding_boxes or draw_labels:
                    hand_label = self.classify_hand(hand_id, img, x_list, y_list, draw_bounding_boxes)
                else:
                    hand_label = self.classify_hand(hand_id, img, x_list, y_list, False)

                all_hands.append({"hand": hand_label, "lm_list": lm_list})
        return all_hands

    def classify_hand(self, hand_id, img, x_list=[], y_list=[], draw_bounding_box=True):
        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)
        bbox = (xmin, ymin, xmax, ymax)

        hand_label = self.results.multi_handedness[hand_id].classification[0].label

        if draw_bounding_box:
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
            cv2.putText(img, hand_label, (xmin - 20, ymin - 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return hand_label
        
    def fingers_up(self, hand):
        lm_list = hand["lm_list"]
        hand_label = hand["hand"] 
        fingers = []

        # Thumb
        if hand_label == "Left":
            fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)
        else:  # Right
            fingers.append(1 if lm_list[4][1] < lm_list[3][1] else 0)

        # 4 Fingers
        tip_ids = [8, 12, 16, 20]
        for tip in tip_ids:
            fingers.append(1 if lm_list[tip][2] < lm_list[tip - 2][2] else 0)

        return fingers


import numpy as np  # make sure this is at the top

def main():
    cap = cv2.VideoCapture(0)

    p_time = 0
    detector = HandDetector()

    # toggle states
    show_bboxes = True
    show_labels = True
    show_landmarks = True

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.find_hands(img, draw_landmarks=show_landmarks)
        all_hands = detector.find_position(img,
                                         draw_labels=show_labels,
                                         draw_bounding_boxes=show_bboxes)
        
        if len(all_hands) > 0:
            hand = all_hands[0]
            fingers = detector.fingers_up(hand)
            print(fingers)

        # FPS calculation
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # ui
        ui_img = 255 * np.ones((300, 600, 3), dtype=np.uint8)

        ui = [
            f"FPS: {int(fps)}",
            "Controls:",
            "q - quit",
            f"b - toggle bounding boxes ({'ON' if show_bboxes else 'OFF'})",
            f"l - toggle landmark labels ({'ON' if show_labels else 'OFF'})",
            f"h - toggle landmarks ({'ON' if show_landmarks else 'OFF'})"
        ]

        for i, text in enumerate(ui):
            cv2.putText(ui_img, text, (15, 40 + i * 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        # Show both windows
        cv2.imshow("Hand Detection", img)
        cv2.imshow("Controls", ui_img)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            show_bboxes = not show_bboxes
        elif key == ord('l'):
            show_labels = not show_labels
        elif key == ord('h'):
            show_landmarks = not show_landmarks

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
