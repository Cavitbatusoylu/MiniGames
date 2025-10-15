try:
    import keyboard as kb
    KEYBOARD_AVAILABLE = True
except Exception:
    kb = None
    KEYBOARD_AVAILABLE = False
import cv2
import numpy as np
import mediapipe as mp
import threading
from time import sleep
import os
import ctypes

# Flip camera frame horizontally if True
FLIP_CAMERA = True
HEAD_CONTROL = True  # Set True to use head movement instead of hand gestures
# Daha hassas algılama için piksel tabanlı alt sınır eşikleri
HEAD_THRESH_X = 6    # pixels (daha düşük = daha hassas)
HEAD_THRESH_Y = 6    # pixels (daha düşük = daha hassas)
# Çözünürlüğe göre dinamik eşik (görüntü genişlik/yüksekliğinin yüzdesi)
HEAD_THRESH_FRAC_X = 0.03  # 3% of frame width
HEAD_THRESH_FRAC_Y = 0.03  # 3% of frame height
# Kafa merkezi için yumuşatma katsayısı (0: yok, 1: tam takip). 0.3-0.5 arası önerilir
HEAD_SMOOTHING_ALPHA = 0.7  # daha hızlı tepki için arttırıldı
HEAD_AXIS_STICKY = 0.0  # eksen histerezisi kapalı -> daha duyarlı eksen geçişi

# Yüz merkez kalibrasyonu ve yön teyidi ayarları
HEAD_BASE_ADAPT_ALPHA = 0.05  # nötr bölgede tabana yumuşak adaptasyon
HEAD_PERSIST_FRAMES = 2       # aynı yön üst üste bu kadar karede teyit edilince uygula
HEAD_CALIBRATE_KEY = 'c'      # kalibrasyon için kısayol (kamera penceresi aktifken)

# Normalleştirilmiş eşikler (yüz kutusu boyutuna göre)
HEAD_THRESH_NORM_X = 0.06  # daha düşük = daha hassas
HEAD_THRESH_NORM_Y = 0.06  # daha düşük = daha hassas
HEAD_PERSIST_FRAMES_X = 1
HEAD_PERSIST_FRAMES_Y = 1
HEAD_Y_THR_MULT = 0.9   # düşürüldü -> dikey yönler biraz daha kolay tetiklenir
HEAD_Y_BIAS = 1.0       # eksenler eşit öncelik

# Koridorda sürtünmeyi azaltmak için otomatik merkezleme
AUTO_ALIGN = True
ALIGN_LOOKAHEAD = 16  # biraz daha erken hizalanma için artırıldı
ALIGN_MAX_NUDGE = 2   # mikro itme büyütüldü

# ------------------------
# Yardımcı fonksiyonlar
# ------------------------
def apply_ema(prev_val, new_val, alpha):
    if prev_val is None:
        return int(new_val)
    return int(alpha * new_val + (1 - alpha) * prev_val)

def compute_dynamic_thresholds(box_w, box_h):
    thrx = max(HEAD_THRESH_X, int(box_w * HEAD_THRESH_NORM_X))
    thry = int(max(HEAD_THRESH_Y, int(box_h * HEAD_THRESH_NORM_Y)) * HEAD_Y_THR_MULT)
    return thrx, thry

def update_persistence(direction):
    global _HEAD_DIR_BUF_X, _HEAD_DIR_BUF_Y
    try:
        _ = _HEAD_DIR_BUF_X
    except NameError:
        _HEAD_DIR_BUF_X = []
        _HEAD_DIR_BUF_Y = []

    final_dir = None
    if direction in ('left', 'right'):
        _HEAD_DIR_BUF_X.append(direction)
        _HEAD_DIR_BUF_Y.clear()
        if len(_HEAD_DIR_BUF_X) > HEAD_PERSIST_FRAMES_X:
            _HEAD_DIR_BUF_X = _HEAD_DIR_BUF_X[-HEAD_PERSIST_FRAMES_X:]
        if len(_HEAD_DIR_BUF_X) == HEAD_PERSIST_FRAMES_X and all(d == direction for d in _HEAD_DIR_BUF_X):
            final_dir = direction
            _HEAD_DIR_BUF_X.clear()
    elif direction in ('up', 'down'):
        _HEAD_DIR_BUF_Y.append(direction)
        _HEAD_DIR_BUF_X.clear()
        if len(_HEAD_DIR_BUF_Y) > HEAD_PERSIST_FRAMES_Y:
            _HEAD_DIR_BUF_Y = _HEAD_DIR_BUF_Y[-HEAD_PERSIST_FRAMES_Y:]
        if len(_HEAD_DIR_BUF_Y) == HEAD_PERSIST_FRAMES_Y and all(d == direction for d in _HEAD_DIR_BUF_Y):
            final_dir = direction
            _HEAD_DIR_BUF_Y.clear()
    else:
        _HEAD_DIR_BUF_X.clear()
        _HEAD_DIR_BUF_Y.clear()

    return final_dir

def auto_align_position(x, y, xf, yf, img, H, W):
    if not AUTO_ALIGN or (xf == 0 and yf == 0):
        return x, y

    if xf != 0:
        x_look = x + (11 if xf > 0 else -11)
        if 0 <= x_look < W:
            top_block = y - 9 >= 0 and np.any(img[y-9, max(0, x-8):min(W, x+8)] == 255)
            bottom_block = y + 9 < H and np.any(img[y+9, max(0, x-8):min(W, x+8)] == 255)
            if top_block and not bottom_block and y + ALIGN_MAX_NUDGE + 9 < H:
                y += ALIGN_MAX_NUDGE
            elif bottom_block and not top_block and y - ALIGN_MAX_NUDGE - 9 >= 0:
                y -= ALIGN_MAX_NUDGE
    elif yf != 0:
        y_look = y + (11 if yf > 0 else -11)
        if 0 <= y_look < H:
            left_block = x - 9 >= 0 and np.any(img[max(0, y-8):min(H, y+8), x-9] == 255)
            right_block = x + 9 < W and np.any(img[max(0, y-8):min(H, y+8), x+9] == 255)
            if left_block and not right_block and x + ALIGN_MAX_NUDGE + 9 < W:
                x += ALIGN_MAX_NUDGE
            elif right_block and not left_block and x - ALIGN_MAX_NUDGE - 9 >= 0:
                x -= ALIGN_MAX_NUDGE
    return x, y

#Image processing and thresholding
#img -> Game matrix (walls = 255, paths = 0)
#img2 -> Maze silhouette for setting playing area (grayscale)

# Load images relative to this script's directory to avoid working directory issues
base_dir = os.path.dirname(__file__)
maze_path = os.path.join(base_dir, 'Maze.jpg')
maze2_path = os.path.join(base_dir, 'Maze2.jpg')

# Read the maze as grayscale and build mask vectorized to match its size
img2 = cv2.imread(maze_path, cv2.IMREAD_GRAYSCALE)
if img2 is None:
    raise FileNotFoundError(f"Maze image not found at: {maze_path}")

img2 = cv2.GaussianBlur(img2, (5, 5), 0)
# Initialize game matrix with walls (255) and carve paths (0) where the maze is dark
img = np.full_like(img2, 255, dtype=np.uint8)
img[img2 < 15] = 0

#Initializations of player
#xf -> x velocity of player
#yf -> y velocity of player
x = 229
y = 183
a = 0
xf = -1
yf = 0
d = 0
score = 0
next_dir = 'right'

#Initializations of food and enemy
food = []
enemy = []
enemyVelocity = []
enemyNext = []

H_init, W_init = img.shape
margin_y, margin_x = 20, 20
while len(enemy) < 5:
    f1 = np.random.randint(margin_y, max(margin_y + 1, H_init - margin_y))
    f2 = np.random.randint(margin_x, max(margin_x + 1, W_init - margin_x))
    #Checking if the enemy is not spawned on the walls
    if np.all(img[f1-10:f1+10, f2-10:f2+10] == 0):
        enemy.append([f2, f1])
        enemyVelocity.append([1, 0])
        enemyNext.append('right')

t = 0
isOver = False
flag = 1

# Gesture mode (hand) setup
if not HEAD_CONTROL:
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = os.path.join(base_dir, 'gesture_recognizer.task')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Gesture model not found at: {model_path}")

    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        global next_dir
        if len(result.gestures):
            cat = result.gestures[0][0].category_name
            if cat != 'none':
                next_dir = cat
            print('Detected gesture:', cat)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result
    )

camstarted = threading.Event()
quit_event = threading.Event()

#Gesture recognition thread
def gestureRecognition():
    global isOver, camstarted

    cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    #Wait for camera to start
    sleep(2)
    camstarted.set()

    if HEAD_CONTROL:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
    else:
        recognizer = GestureRecognizer.create_from_options(options)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            if HEAD_CONTROL:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    face_lm = result.multi_face_landmarks[0]
                    # Yüz kutusu boyutunu landmarks'tan hesapla
                    xs = [int(pt.x * w) for pt in face_lm.landmark]
                    ys = [int(pt.y * h) for pt in face_lm.landmark]
                    min_x, max_x = max(min(xs), 0), min(max(xs), w-1)
                    min_y, max_y = max(min(ys), 0), min(max(ys), h-1)
                    box_w = max(1, max_x - min_x)
                    box_h = max(1, max_y - min_y)

                    # Burun ucu olarak genelde 1. indeks kullanılır; yoksa merkez
                    try:
                        nose_pt = face_lm.landmark[1]
                        cx = int(nose_pt.x * w)
                        cy = int(nose_pt.y * h)
                    except Exception:
                        cx = (min_x + max_x) // 2
                        cy = (min_y + max_y) // 2

                    # Yumuşatma (EMA)
                    global _HEAD_SX, _HEAD_SY
                    try:
                        _ = _HEAD_SX
                    except NameError:
                        _HEAD_SX = None
                        _HEAD_SY = None
                    if _HEAD_SX is None:
                        _HEAD_SX = cx
                        _HEAD_SY = cy
                    else:
                        _HEAD_SX = apply_ema(_HEAD_SX, cx, HEAD_SMOOTHING_ALPHA)
                        _HEAD_SY = apply_ema(_HEAD_SY, cy, HEAD_SMOOTHING_ALPHA)

                    cx_s = _HEAD_SX
                    cy_s = _HEAD_SY

                    # Kalibrasyon merkezi (nötr poz) ve yön teyidi dizileri
                    global _HEAD_BASE_X, _HEAD_BASE_Y, _HEAD_DIR_BUFFER
                    try:
                        _ = _HEAD_BASE_X
                    except NameError:
                        _HEAD_BASE_X = w // 2
                        _HEAD_BASE_Y = h // 2
                        _HEAD_DIR_BUFFER = []

                    # Manuel kalibrasyon: 'c' basılırsa mevcut kafa merkezini baz al
                    if cv2.waitKey(1) & 0xFF == ord(HEAD_CALIBRATE_KEY):
                        _HEAD_BASE_X = cx_s
                        _HEAD_BASE_Y = cy_s

                    # Nötr bölgede yavaşça baza doğru adaptasyon (drift'i azaltır)
                    if abs(cx_s - _HEAD_BASE_X) < 8 and abs(cy_s - _HEAD_BASE_Y) < 8:
                        _HEAD_BASE_X = int(HEAD_BASE_ADAPT_ALPHA * cx_s + (1 - HEAD_BASE_ADAPT_ALPHA) * _HEAD_BASE_X)
                        _HEAD_BASE_Y = int(HEAD_BASE_ADAPT_ALPHA * cy_s + (1 - HEAD_BASE_ADAPT_ALPHA) * _HEAD_BASE_Y)

                    # Draw for visual feedback
                    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)
                    cv2.circle(frame, (cx_s, cy_s), 4, (0, 0, 255), -1)

                    # Compare against frame center
                    dx = cx_s - _HEAD_BASE_X
                    dy = cy_s - _HEAD_BASE_Y

                    # Dinamik eşikler: yüz kutusu boyutuna göre (mesafeye dayanıklı)
                    thrx, thry = compute_dynamic_thresholds(box_w, box_h)

                    # Decide direction using thresholds
                    # Histerezis: son ekseni korumaya çalış
                    global _HEAD_LAST_AXIS
                    try:
                        _ = _HEAD_LAST_AXIS
                    except NameError:
                        _HEAD_LAST_AXIS = None

                    prefer_x = abs(dx) * (1 + (HEAD_AXIS_STICKY if _HEAD_LAST_AXIS == 'x' else 0))
                    prefer_y = abs(dy) * (HEAD_Y_BIAS) * (1 + (HEAD_AXIS_STICKY if _HEAD_LAST_AXIS == 'y' else 0))

                    if prefer_x > prefer_y:
                        if dx > thrx:
                            direction = 'right'
                        elif dx < -thrx:
                            direction = 'left'
                        else:
                            direction = None
                    else:
                        if dy > thry:
                            direction = 'down'
                        elif dy < -thry:
                            direction = 'up'
                        else:
                            direction = None

                    # Yön teyidi: X ve Y için ayrı kalıcılık (persistency)
                    global _HEAD_DIR_BUF_X, _HEAD_DIR_BUF_Y
                    try:
                        _ = _HEAD_DIR_BUF_X
                    except NameError:
                        _HEAD_DIR_BUF_X = []
                        _HEAD_DIR_BUF_Y = []

                    final_dir = update_persistence(direction)

                    if final_dir in ('left', 'right'):
                        _HEAD_LAST_AXIS = 'x'
                        globals()['next_dir'] = final_dir
                    elif final_dir in ('up', 'down'):
                        _HEAD_LAST_AXIS = 'y'
                        globals()['next_dir'] = final_dir
                # else: keep previous direction
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                recognizer.recognize_async(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

            cv2.imshow('Gesture Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_event.set()
                break
    finally:
        cap.release()
        if not HEAD_CONTROL:
            recognizer.close()
        else:
            face_mesh.close()
        cv2.destroyAllWindows()

#Game thread
def game():
    global isOver, t, next_dir, x, y, xf, yf, d, a, flag, score, food, enemy, enemyVelocity, enemyNext, img, camstarted
    
    # Fullscreen config (Windows)
    try:
        SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
        SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        SCREEN_W, SCREEN_H = 1280, 720
    window_inited = False
    
    #Wait for recognition to start
    camstarted.wait()

    # Load maze frame once and reuse each iteration for performance
    base_maze = cv2.imread(maze2_path)
    if base_maze is None:
        raise FileNotFoundError(f"Maze2 image not found at: {maze2_path}")

    H, W = img.shape

    while not quit_event.is_set():

        #Image for maze to be displayed (copy from cached base image)
        img3 = base_maze.copy()

        #Mouth opening angle of pacman
        if a <= 0:
            flag = 1
        if a >= 30:
            flag = 0
        if flag:
            a += 1
        else:
            a -= 1

        #Spawning food
        while len(food) < 10:
            f1 = np.random.randint(margin_y, max(margin_y + 1, H - margin_y))
            f2 = np.random.randint(margin_x, max(margin_x + 1, W - margin_x))
            #Checking if the food is not spawned on the walls
            if np.all(img[f1-10:f1+10, f2-10:f2+10] == 0):
                food.append([f2, f1])
                
        for x1, y1 in food:
            cv2.rectangle(img3, (x1-5, y1-5), (x1+5, y1+5), (255, 255, 0), -1)
            #Detecting collision with food
            if x1-5 <= x <= x1+5 and y1-5 <= y <= y1+5:
                food.remove([x1, y1])
                score += 1

        for i in range(len(enemy)):
            #Randomly changing direction of enemy on every 250th iteration
            #This ensures randomness in motion and avoids getting stuck in a corner
            if t % 250 == 0:
                enemyNext[i] = np.random.choice(['up', 'down', 'left', 'right'])
            if enemyVelocity[i] == [0, 0]:
                enemyNext[i] = np.random.choice(['up', 'down', 'left', 'right'])

            x1 = enemy[i][0]
            y1 = enemy[i][1]

            #Set velocity of enemy based on direction and feasibility of movement
            if enemyNext[i] == 'right' and np.all(img[y1-7:y1+7, x1+9] != 255):
                enemyVelocity[i] = [1, 0]
            elif enemyNext[i] == 'left' and np.all(img[y1-7:y1+7, x1-9] != 255):
                enemyVelocity[i] = [-1, 0]
            elif enemyNext[i] == 'down' and np.all(img[y1+9, x1-7:x1+7] != 255):
                enemyVelocity[i] = [0, 1]
            elif enemyNext[i] == 'up' and np.all(img[y1-9, x1-7:x1+7] != 255):
                enemyVelocity[i] = [0, -1]

            #If motion is not feasible, stop for a moment and change direction
            if enemyVelocity[i][0] == 1 and np.any(img[y1-7:y1+7, x1+9] == 255):
                enemyVelocity[i] = [0, 0]
                enemyNext[i] = np.random.choice(['up', 'down', 'left'])
            elif enemyVelocity[i][0] == -1 and np.any(img[y1-7:y1+7, x1-9] == 255):
                enemyVelocity[i] = [0, 0]
                enemyNext[i] = np.random.choice(['up', 'down', 'right'])
            elif enemyVelocity[i][1] == 1 and np.any(img[y1+9, x1-7:x1+7] == 255):
                enemyVelocity[i] = [0, 0]
                enemyNext[i] = np.random.choice(['up', 'left', 'right'])
            elif enemyVelocity[i][1] == -1 and np.any(img[y1-9, x1-7:x1+7] == 255):
                enemyVelocity[i] = [0, 0]
                enemyNext[i] = np.random.choice(['down', 'left', 'right'])

            #Enemy motion
            enemy[i][0] = int(round(x1 + enemyVelocity[i][0]))
            enemy[i][1] = int(round(y1 + enemyVelocity[i][1]))
            x1 = enemy[i][0]
            y1 = enemy[i][1]
            cv2.circle(img3, (x1, y1), 6, (255, 0, 255), -1)

            #Detecting collision with enemy
            if x1-5 <= x <= x1+5 and y1-5 <= y <= y1+5:
                isOver = True

        #Scoreboard
        if isOver != True:
            cv2.ellipse(img3, (x, y), (7, 7), d, a, 360-(2*a), (0, 0, 255), -1)
            cv2.putText(img3, "Score: " + str(score), (217, 237), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(img3, "Game Over", (210, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(img3, "Score: " + str(score), (217, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)    
            # Short pause, then reset the game state for endless play
            cv2.imshow("Pacman", img3)
            cv2.waitKey(800)
            # Reset state
            x = 229
            y = 183
            a = 0
            xf = -1
            yf = 0
            d = 0
            score = 0
            next_dir = 'right'
            food = []
            enemy = []
            enemyVelocity = []
            enemyNext = []
            # Respawn enemies
            while len(enemy) < 5:
                f1 = np.random.randint(margin_y, max(margin_y + 1, H - margin_y))
                f2 = np.random.randint(margin_x, max(margin_x + 1, W - margin_x))
                if np.all(img[f1-10:f1+10, f2-10:f2+10] == 0):
                    enemy.append([f2, f1])
                    enemyVelocity.append([1, 0])
                    enemyNext.append('right')
            isOver = False

        #Keyboard input for pacman motion (if keyboard module available)
        if KEYBOARD_AVAILABLE:
            if kb.is_pressed('up'):
                next_dir = "up"
            if kb.is_pressed('down'):
                next_dir = "down"
            if kb.is_pressed('left'):
                next_dir = "left"
            if kb.is_pressed('right'):
                next_dir = "right"
        
        #Set velocity of pacman based on direction and feasibility of movement
        def sgn(val):
            return 1 if val > 0 else (-1 if val < 0 else 0)

        # Base feasibility checks at current center
        can_up = y - 16 >= 0 and x - 9 >= 0 and x + 9 < W and np.all(img[y-16, x-9:x+9] != 255)
        can_down = y + 16 < H and x - 9 >= 0 and x + 9 < W and np.all(img[y+16, x-9:x+9] != 255)
        can_left = x - 16 >= 0 and y - 9 >= 0 and y + 9 < H and np.all(img[y-9:y+9, x-16] != 255)
        can_right = x + 16 < W and y - 9 >= 0 and y + 9 < H and np.all(img[y-9:y+9, x+16] != 255)

        # Look-ahead feasibility to allow turning slightly before perfect center
        fwd_px = 18  # kavşaklarda daha erken dönüş için artırıldı
        can_up_fwd = False
        can_down_fwd = False
        can_left_fwd = False
        can_right_fwd = False
        if xf != 0:
            x2 = x + sgn(xf) * fwd_px
            if x2 - 9 >= 0 and x2 + 9 < W:
                if y - 16 >= 0:
                    can_up_fwd = np.all(img[y-16, x2-9:x2+9] != 255)
                if y + 16 < H:
                    can_down_fwd = np.all(img[y+16, x2-9:x2+9] != 255)
        if yf != 0:
            y2 = y + sgn(yf) * fwd_px
            if y2 - 9 >= 0 and y2 + 9 < H:
                if x - 16 >= 0:
                    can_left_fwd = np.all(img[y2-9:y2+9, x-16] != 255)
                if x + 16 < W:
                    can_right_fwd = np.all(img[y2-9:y2+9, x+16] != 255)

        if next_dir == 'up' and (can_up or can_up_fwd):
            yf = -1
            xf = 0
        if next_dir == 'down' and (can_down or can_down_fwd):
            yf = 1
            xf = 0
        if next_dir == 'left' and (can_left or can_left_fwd):
            xf = -1
            yf = 0
        if next_dir == 'right' and (can_right or can_right_fwd):
            xf = 1
            yf = 0
        
        #Update position of pacman and direction of mouth opening
        if xf > 0:
            x += 1
            d = 0
        elif xf < 0:
            x -= 1
            d = 180
        if yf > 0:
            y += 1
            d = 90
        elif yf < 0:
            y -= 1
            d = -90

        # Koridor merkezleme: yardımcı fonksiyon ile
        x, y = auto_align_position(x, y, xf, yf, img, H, W)

        #Stop pacman if motion is not feasible; try corner turn into desired direction
        hit_wall = False
        if xf == 1 and x + 11 < W and y - 9 >= 0 and y + 9 < H and np.any(img[y-9:y+9, x+11] == 255):
            xf = 0
            hit_wall = True
        if xf == -1 and x - 11 >= 0 and y - 9 >= 0 and y + 9 < H and np.any(img[y-9:y+9, x-11] == 255):
            xf = 0
            hit_wall = True
        if yf == 1 and y + 11 < H and x - 9 >= 0 and x + 9 < W and np.any(img[y+11, x-9:x+9] == 255):
            yf = 0
            hit_wall = True
        if yf == -1 and y - 11 >= 0 and x - 9 >= 0 and x + 9 < W and np.any(img[y-11, x-9:x+9] == 255):
            yf = 0
            hit_wall = True

        if hit_wall:
            # Recompute feasibility at current position after collision
            can_up2 = y - 16 >= 0 and x - 9 >= 0 and x + 9 < W and np.all(img[y-16, x-9:x+9] != 255)
            can_down2 = y + 16 < H and x - 9 >= 0 and x + 9 < W and np.all(img[y+16, x-9:x+9] != 255)
            can_left2 = x - 16 >= 0 and y - 9 >= 0 and y + 9 < H and np.all(img[y-9:y+9, x-16] != 255)
            can_right2 = x + 16 < W and y - 9 >= 0 and y + 9 < H and np.all(img[y-9:y+9, x+16] != 255)

            # On collision, immediately try turning to the requested direction if possible
            if next_dir == 'up' and can_up2:
                yf = -1
                xf = 0
            elif next_dir == 'down' and can_down2:
                yf = 1
                xf = 0
            elif next_dir == 'left' and can_left2:
                xf = -1
                yf = 0
            elif next_dir == 'right' and can_right2:
                xf = 1
                yf = 0

        t += 1
        if not window_inited:
            cv2.namedWindow("Pacman", cv2.WINDOW_NORMAL)
            # Başlangıçta makul bir pencere boyutu ayarla (yeniden boyutlandırılabilir)
            init_w = min(SCREEN_W, 900)
            init_h = min(SCREEN_H, 900)
            try:
                cv2.resizeWindow("Pacman", init_w, init_h)
            except Exception:
                pass
            window_inited = True
        # Mevcut pencere boyutuna göre ölçekle
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect("Pacman")
            if win_w > 0 and win_h > 0:
                display = cv2.resize(img3, (win_w, win_h), interpolation=cv2.INTER_NEAREST)
            else:
                display = img3
        except Exception:
            display = img3
        cv2.imshow("Pacman", display)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            quit_event.set()
        # Pencere kapatılırsa çık
        try:
            if cv2.getWindowProperty("Pacman", cv2.WND_PROP_VISIBLE) < 1:
                quit_event.set()
        except Exception:
            pass

if __name__ == "__main__":
    #Threading for concurrent execution of gesture recognition and game
    gesture_thread = threading.Thread(target=gestureRecognition)
    game_thread = threading.Thread(target=game)

    gesture_thread.start()
    game_thread.start()

    gesture_thread.join()
    game_thread.join()

    cv2.destroyAllWindows()