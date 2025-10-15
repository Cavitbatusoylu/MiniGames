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
DOWN_DEBOUNCE_MS = 200  # Duck debounce
OPEN_DELTA_Y = 15   # index tip above PIP by this -> consider open
CLOSE_DELTA_Y = 10  # index tip below PIP by this -> consider closed
DUCK_OPEN_DELTA_Y = 25  # thumb tip above IP by this -> consider not ducking
DUCK_CLOSE_DELTA_Y = 15  # thumb tip below IP by this -> consider ducking


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
    # Try different camera indices
    cap = None
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"Camera {camera_index} opened successfully!")
                break
            else:
                cap.release()
                cap = None
        else:
            cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        print("No camera could be opened. Please check:")
        print("1. Camera is connected and not used by another application")
        print("2. Camera permissions are granted")
        print("3. Try closing other camera applications (Zoom, Skype, etc.)")
        hands.close()
        return
    
    # Set capture resolution with error handling
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        # Verify the resolution was set
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution set to: {actual_width}x{actual_height}")
    except Exception as e:
        print(f"Warning: Could not set camera resolution: {e}")
        print("Continuing with default resolution...")

    last_space_time = 0.0
    last_down_time = 0.0
    down_pressed = False
    index_open_prev = False
    thumb_down_prev = False
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
                        # Thumb down to duck with hysteresis and debounce
                        thumb_tip_y = kpos[THUMB_TIP_ID][1]
                        thumb_ip_y = kpos[THUMB_IP_ID][1]
                        
                        if not thumb_down_prev:
                            # was not ducking; require stronger duck condition to switch to ducking
                            thumb_down = thumb_tip_y + DUCK_CLOSE_DELTA_Y > thumb_ip_y
                        else:
                            # was ducking; require stronger not-duck condition to switch to not ducking
                            thumb_down = not (thumb_tip_y + DUCK_OPEN_DELTA_Y < thumb_ip_y)
                        
                        if thumb_down and not thumb_down_prev:
                            # rising edge: just started ducking
                            if now_ms - last_down_time > DOWN_DEBOUNCE_MS:
                                if not down_pressed:
                                    pyautogui.keyDown('down')
                                    down_pressed = True
                                    last_down_time = now_ms
                        elif not thumb_down and thumb_down_prev:
                            # falling edge: just stopped ducking
                            if down_pressed:
                                pyautogui.keyUp('down')
                                down_pressed = False
                        
                        thumb_down_prev = thumb_down

            cv2.imshow("OpenCV", frame)
            key = cv2.waitKey(1) & 0xFF
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
    # Disable the start button to prevent multiple clicks
    start.config(state='disabled', text='Starting...')
    gui.update()
    
    # Start detection in a separate thread
    detect_thread = threading.Thread(target=Detect_def, daemon=True)
    detect_thread.start()
    
    # Re-enable button after a short delay
    def re_enable_button():
        time.sleep(2)  # Wait 2 seconds
        start.config(state='normal', text='Start')
        gui.update()
    
    threading.Thread(target=re_enable_button, daemon=True).start()


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
    