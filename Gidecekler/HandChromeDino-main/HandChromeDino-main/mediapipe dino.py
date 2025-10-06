import tkinter as tk
import mediapipe as mp
import pyautogui
import threading
import cv2
import time

gui = tk.Tk()
gui.title("Python Mediapipe Chrome Dino")
gui.geometry("800x600")
gui.resizable(0, 0)


# Config
FLIP_CAMERA = True
# Capture/display resolution (try lower values for speed)
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Processing resolution (smaller = faster). Landmarks are normalized, so scaling is fine
PROC_WIDTH = 448
PROC_HEIGHT = 252

# Gesture thresholds (in pixels in processed frame coordinates)
INDEX_TIP_ID = 8
INDEX_PIP_ID = 6
THUMB_TIP_ID = 4
THUMB_IP_ID = 3

JUMP_DELTA_Y = 20  # tip significantly lower than PIP means closed finger (custom heuristic)
DUCK_DELTA_Y = 20  # thumb tip lower than IP means thumbs-down (custom heuristic)

# Debounce/hysteresis
SPACE_DEBOUNCE_MS = 250
OPEN_DELTA_Y = 15   # index tip above PIP by this -> consider open
CLOSE_DELTA_Y = 10  # index tip below PIP by this -> consider closed


def Detect_def():
    print("Start Detecting ... ")
    mp_hand = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hand.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,           # lighter model for speed
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera could not be opened")
        hands.close()
        return
    # Try set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    last_space_time = 0.0
    down_pressed = False
    index_open_prev = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc = cv2.resize(rgb, (PROC_WIDTH, PROC_HEIGHT))
            results = hands.process(proc)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on display frame by reusing connections (optional to save time)
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        mp_hand.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    kpos = []
                    for i in range(21):
                        x = hand_landmarks.landmark[i].x * PROC_WIDTH
                        y = hand_landmarks.landmark[i].y * PROC_HEIGHT
                        kpos.append((x, y))

                    # Key actions with state + debounce
                    now_ms = time.time() * 1000.0
                    if len(kpos) > max(INDEX_TIP_ID, INDEX_PIP_ID):
                        # Determine index open/closed with hysteresis (open triggers jump on rising edge only)
                        index_tip_y = kpos[INDEX_TIP_ID][1]
                        index_pip_y = kpos[INDEX_PIP_ID][1]
                        if not index_open_prev:
                            # was closed; require stronger open condition to switch to open
                            index_open = index_tip_y + OPEN_DELTA_Y < index_pip_y
                        else:
                            # was open; require stronger close condition to switch to closed
                            index_open = not (index_tip_y + CLOSE_DELTA_Y > index_pip_y)

                        if (not index_open_prev) and index_open:
                            # rising edge: just opened -> jump once
                            if now_ms - last_space_time > SPACE_DEBOUNCE_MS:
                                pyautogui.press('space')
                                last_space_time = now_ms
                        index_open_prev = index_open

                    if len(kpos) > max(THUMB_TIP_ID, THUMB_IP_ID):
                        # Thumb down to duck
                        if kpos[THUMB_TIP_ID][1] + DUCK_DELTA_Y > kpos[THUMB_IP_ID][1]:
                            if not down_pressed:
                                pyautogui.keyDown('down')
                                down_pressed = True
                        else:
                            if down_pressed:
                                pyautogui.keyUp('down')
                                down_pressed = False

            cv2.imshow("OpenCV", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        # Ensure keys are released and resources are cleaned up
        try:
            pyautogui.keyUp('down')
        except Exception:
            pass
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


def start_detect():
    detect_thread = threading.Thread(target=Detect_def, daemon=True)
    detect_thread.start()


# GUI Tasarımı
title_frame = tk.Frame(gui)
title_frame.pack()
title = tk.Label(title_frame,
                 text="Python Chrome Dino\n",
                 font=("Times New Roman", 36))
title_frame.place(x=200, y=100)
title.pack()

start_frame = tk.Frame()
start_frame.pack()

start = tk.Button(start_frame,
                  text="Start",
                  font=("Times New Roman", 36),
                  bg="pink",
                  command=start_detect)
start_frame.place(x=450, y=350)
start.pack()

package_frame = tk.Frame()
package_frame.pack()

package = tk.Label(package_frame,
                   text='  Version and Packages：\n Python 3.9\n Mediapipe \n PyAutoGUI\n OpenCV\n Tkinter',
                   font=("Times New Roman", 20))

package.pack()
package_frame.place(x=50, y=300)

gui.mainloop()
    