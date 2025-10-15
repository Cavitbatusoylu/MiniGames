import cv2
import mediapipe as mp
import pygame
import random
import sys
import math
from collections import deque
import time

# --- Pygame Ayarları ---
pygame.init()
SMALL_W, SMALL_H = 1000, 800
info = pygame.display.Info()
FULL_W, FULL_H = info.current_w, info.current_h

is_fullscreen = False
WIDTH, HEIGHT = SMALL_W, SMALL_H
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Galactic Face Commander")
pygame.mouse.set_visible(False)

clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)
big_font = pygame.font.Font(None, 72)

# Renkler
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
GOLD = (255, 215, 0)

# --- Parçacık Sistemi ---
class Particle:
    def __init__(self, x, y, color, velocity, life, size=3):
        self.x = x
        self.y = y
        self.color = color
        self.vx, self.vy = velocity
        self.life = life
        self.max_life = life
        self.size = size
        self.gravity = random.uniform(0, 0.2)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.vx *= 0.98
        self.vy *= 0.98
        self.life -= 1
        self.size = max(0, self.size * 0.95)
        return self.life > 0
    
    def draw(self, screen):
        if self.size > 0:
            alpha = int(255 * (self.life / self.max_life))
            s = pygame.Surface((self.size*2, self.size*2))
            s.set_alpha(alpha)
            pygame.draw.circle(s, self.color, (self.size, self.size), int(self.size))
            screen.blit(s, (self.x - self.size, self.y - self.size))

particles = []

# --- Power-ups ---
class PowerUp:
    def __init__(self, x, y, power_type):
        self.rect = pygame.Rect(x, y, 30, 30)
        self.type = power_type  # 'triple', 'rapid', 'shield', 'health'
        self.colors = {
            'triple': CYAN,
            'rapid': YELLOW,
            'shield': PURPLE,
            'health': GREEN
        }
        self.glow = 0
        self.glow_dir = 1
    
    def update(self):
        self.rect.y += 2
        self.glow += self.glow_dir * 5
        if self.glow >= 50 or self.glow <= 0:
            self.glow_dir *= -1
        return self.rect.y < HEIGHT
    
    def draw(self, screen):
        color = self.colors[self.type]
        # Glow efekti
        for i in range(3):
            glow_color = [min(255, c + self.glow) for c in color]
            pygame.draw.circle(screen, glow_color, self.rect.center, 20 - i*5, 2)
        # Ana power-up
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)

# --- Düşman Sınıfları ---
class Enemy:
    def __init__(self, x, y, enemy_type='basic'):
        self.rect = pygame.Rect(x, y, 40, 40)
        self.type = enemy_type
        self.health = {'basic': 1, 'fast': 1, 'tank': 3, 'zigzag': 2}[enemy_type]
        self.max_health = self.health
        self.speed = {'basic': 3, 'fast': 6, 'tank': 1, 'zigzag': 4}[enemy_type]
        self.colors = {'basic': RED, 'fast': ORANGE, 'tank': PURPLE, 'zigzag': CYAN}
        self.direction = random.choice([-1, 1])
        self.zigzag_timer = 0
        self.shoot_timer = random.randint(60, 120)
    
    def update(self):
        if self.type == 'zigzag':
            self.zigzag_timer += 1
            if self.zigzag_timer % 30 == 0:
                self.direction *= -1
            self.rect.x += self.direction * 2
        
        self.rect.y += self.speed
        
        # Kenarlardan sekme
        if self.rect.x < 0 or self.rect.x > WIDTH - self.rect.width:
            self.direction *= -1
        
        self.shoot_timer -= 1
        return self.rect.y < HEIGHT
    
    def draw(self, screen):
        color = self.colors[self.type]
        # Gölge
        shadow = self.rect.copy()
        shadow.x += 2
        shadow.y += 2
        pygame.draw.rect(screen, (50, 50, 50), shadow)
        
        # Ana düşman
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        
        # Sağlık çubuğu
        if self.health < self.max_health:
            health_bar = pygame.Rect(self.rect.x, self.rect.y - 10, self.rect.width, 5)
            pygame.draw.rect(screen, RED, health_bar)
            health_fill = pygame.Rect(self.rect.x, self.rect.y - 10, 
                                    int(self.rect.width * (self.health / self.max_health)), 5)
            pygame.draw.rect(screen, GREEN, health_fill)

# --- Boss Sınıfı ---
class Boss:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH//2 - 60, 50, 120, 80)
        self.health = 50
        self.max_health = 50
        self.direction = 1
        self.shoot_timer = 0
        self.pattern = 0
        self.pattern_timer = 0
    
    def update(self):
        self.rect.x += self.direction * 2
        if self.rect.x < 0 or self.rect.x > WIDTH - self.rect.width:
            self.direction *= -1
        
        self.shoot_timer -= 1
        self.pattern_timer += 1
        
        if self.pattern_timer > 180:
            self.pattern = (self.pattern + 1) % 3
            self.pattern_timer = 0
        
        return self.health > 0
    
    def draw(self, screen):
        # Boss gölgesi
        shadow = self.rect.copy()
        shadow.x += 4
        shadow.y += 4
        pygame.draw.rect(screen, (50, 50, 50), shadow)
        
        # Boss (yanıp sönen renk)
        flash = int(time.time() * 10) % 2
        color = (255, 50, 50) if flash else (200, 0, 0)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 3)
        
        # Sağlık çubuğu
        health_bar = pygame.Rect(50, 20, WIDTH - 100, 20)
        pygame.draw.rect(screen, RED, health_bar)
        health_fill = pygame.Rect(50, 20, int((WIDTH - 100) * (self.health / self.max_health)), 20)
        pygame.draw.rect(screen, GREEN, health_fill)
        
        # Boss adı
        boss_text = font.render("GALACTIC DESTROYER", True, WHITE)
        screen.blit(boss_text, (WIDTH//2 - boss_text.get_width()//2, 45))

# --- Mermi Sınıfları ---
class Bullet:
    def __init__(self, x, y, direction=1, bullet_type='normal'):
        self.rect = pygame.Rect(x, y, 8, 15)
        self.direction = direction
        self.type = bullet_type
        self.speed = 15 if direction == -1 else 10
        self.trail = deque(maxlen=6)
        self.glow = 0
        self.glow_dir = 1
    
    def update(self):
        self.trail.append((self.rect.centerx, self.rect.centery))
        self.rect.y += self.speed * self.direction
        self.glow += self.glow_dir * 10
        if self.glow >= 100 or self.glow <= 0:
            self.glow_dir *= -1
        return 0 <= self.rect.y <= HEIGHT
    
    def draw(self, screen):
        # İz efekti
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)))
            size = int(5 * (i / len(self.trail)))
            if size > 0:
                base_color = YELLOW if self.direction == -1 else CYAN
                glow_color = [min(255, c + self.glow//4) for c in base_color]
                pygame.draw.circle(screen, glow_color, (tx, ty), size)
        
        # Ana mermi (parlak efekti)
        base_color = YELLOW if self.direction == -1 else CYAN
        glow_color = [min(255, c + self.glow//2) for c in base_color]
        pygame.draw.rect(screen, glow_color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 1)

# --- Oyuncu Sınıfı ---
class Player:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH//2 - 20, HEIGHT - 80, 40, 60)
        self.health = 100
        self.max_health = 100
        self.shield = 0
        self.weapon_type = 'normal'
        self.weapon_timer = 0
        self.invulnerable = 0
        self.trail = deque(maxlen=8)
    
    def update(self):
        self.trail.append((self.rect.centerx, self.rect.centery))
        if self.weapon_timer > 0:
            self.weapon_timer -= 1
        if self.invulnerable > 0:
            self.invulnerable -= 1
    
    def take_damage(self, damage):
        if self.invulnerable > 0:
            return False
        
        if self.shield > 0:
            self.shield -= damage
            if self.shield < 0:
                self.health += self.shield
                self.shield = 0
        else:
            self.health -= damage
        
        self.invulnerable = 60
        
        # Hasar parçacıkları
        for _ in range(10):
            particle = Particle(
                self.rect.centerx, self.rect.centery,
                RED, (random.uniform(-5, 5), random.uniform(-5, 5)),
                30, 4
            )
            particles.append(particle)
        
        return self.health <= 0
    
    def draw(self, screen):
        # Oyuncu izi
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)))
            size = int(20 * (i / len(self.trail)))
            if size > 0:
                color = [int(c * (i / len(self.trail))) for c in GREEN]
                pygame.draw.circle(screen, color, (tx, ty), size)
        
        # Kalkan efekti
        if self.shield > 0:
            for i in range(3):
                shield_color = [100, 150, 255, 100 - i*30]
                pygame.draw.circle(screen, shield_color[:3], self.rect.center, 35 + i*5, 3)
        
        # Oyuncu (yanıp sönme efekti)
        if self.invulnerable == 0 or self.invulnerable % 10 < 5:
            # Gölge
            shadow = self.rect.copy()
            shadow.x += 2
            shadow.y += 2
            pygame.draw.rect(screen, (50, 50, 50), shadow)
            
            # Ana oyuncu
            pygame.draw.rect(screen, GREEN, self.rect)
            pygame.draw.rect(screen, WHITE, self.rect, 2)

# --- Yıldız Alanı ---
def create_starfield(num_stars=100):
    stars = []
    for _ in range(num_stars):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        speed = random.uniform(0.5, 3)
        size = random.randint(1, 3)
        brightness = random.randint(100, 255)
        stars.append([x, y, speed, size, brightness])
    return stars

def update_starfield(stars):
    for star in stars:
        star[1] += star[2]
        if star[1] > HEIGHT:
            star[1] = 0
            star[0] = random.randint(0, WIDTH)

def draw_starfield(screen, stars):
    for star in stars:
        color = (star[4], star[4], star[4])
        pygame.draw.circle(screen, color, (int(star[0]), int(star[1])), star[3])

# --- Oyun Değişkenleri ---
player = Player()
bullets = []
enemy_bullets = []
enemies = []
powerups = []
boss = None
stars = create_starfield(150)

score = 0
wave = 1
enemies_killed = 0
last_enemy_spawn = 0
game_state = 'playing'  # 'playing', 'game_over', 'boss_fight'
game_over_time = None

# --- Face Detection ---
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_faces=1
)

cap = cv2.VideoCapture(0)
try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass
SHOW_CAMERA_WINDOW = True  # Kamera penceresini göster
CAM_WIN = "Galactic Face Commander Camera"
if SHOW_CAMERA_WINDOW:
    cv2.namedWindow(CAM_WIN, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(CAM_WIN, 640, 480)
    except Exception:
        pass

# Kontrol yumuşatma
face_smoothing = deque(maxlen=5)
eye_smoothing = deque(maxlen=2)
last_shoot_time = 0
# Blink algısı için başlangıç değerleri
ear_threshold = 0.25  # anlık gösterim için; her karede güncellenecek
fire_debounce_ms = 140
ear_baseline = None
BLINK_PERSIST_FRAMES = 2
closed_frames = 0
fired_this_closure = False
face_angle_ema = None  # kafa eğiminde gecikmeyi azaltmak için EMA

def _dist(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.hypot(dx, dy)

def compute_ear_from_facemesh(landmarks, left=True):
    if left:
        # Sol göz: üst (159, 158), alt (145, 153), yatay (33, 133)
        v1 = _dist(landmarks[159], landmarks[145])
        v2 = _dist(landmarks[158], landmarks[153])
        h = _dist(landmarks[33], landmarks[133])
    else:
        # Sağ göz: üst (386, 385), alt (374, 380), yatay (362, 263)
        v1 = _dist(landmarks[386], landmarks[374])
        v2 = _dist(landmarks[385], landmarks[380])
        h = _dist(landmarks[362], landmarks[263])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def create_explosion(x, y, size='normal'):
    """Patlama efekti oluştur"""
    particle_count = {'small': 15, 'normal': 25, 'large': 40}[size]
    colors = [RED, ORANGE, YELLOW, WHITE]
    
    for _ in range(particle_count):
        color = random.choice(colors)
        velocity = (random.uniform(-8, 8), random.uniform(-8, 8))
        life = random.randint(30, 60)
        particle_size = random.randint(2, 6)
        particle = Particle(x, y, color, velocity, life, particle_size)
        particles.append(particle)

def spawn_enemy():
    """Düşman oluştur"""
    enemy_types = ['basic', 'fast', 'tank', 'zigzag']
    weights = [0.5, 0.3, 0.1, 0.1] if wave < 5 else [0.3, 0.3, 0.2, 0.2]
    enemy_type = random.choices(enemy_types, weights)[0]
    x = random.randint(0, WIDTH - 40)
    enemy = Enemy(x, -40, enemy_type)
    enemies.append(enemy)

def spawn_powerup(x, y):
    """Power-up oluştur"""
    if random.random() < 0.3:  # %30 şans
        power_types = ['triple', 'rapid', 'shield', 'health']
        power_type = random.choice(power_types)
        powerup = PowerUp(x, y, power_type)
        powerups.append(powerup)

def restart_game():
    """Uygulamayı kapatmadan oyunu başa al."""
    global player, bullets, enemy_bullets, enemies, powerups, particles
    global boss, score, wave, enemies_killed, game_state, ear_baseline, game_over_time
    player = Player()
    bullets.clear()
    enemy_bullets.clear()
    enemies.clear()
    powerups.clear()
    particles.clear()
    boss = None
    score = 0
    wave = 1
    enemies_killed = 0
    game_state = 'playing'
    ear_baseline = None
    eye_smoothing.clear()
    game_over_time = None

# Ana oyun döngüsü
while True:
    current_time = pygame.time.get_ticks()
    dt = clock.tick(60) / 1000.0
    # Varsayılan kontrol değerleri (kamera başarısız olsa da güvenli)
    move_dir = 0
    shoot = False
    
    # --- Kamera ve Yüz Algılama ---
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = face_mesh.process(rgb)
        except Exception:
            results = None
        
        move_dir = 0
        shoot = False
        
        if results and results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                landmarks = face.landmark
                
                # Gelişmiş kafa eğimi algılama
                nose_tip = landmarks[1]
                left_cheek = landmarks[234]
                right_cheek = landmarks[454]
                forehead = landmarks[10]
                
                # Kafa eğimi hesaplama
                dx = right_cheek.x - left_cheek.x
                dy = right_cheek.y - left_cheek.y
                angle = math.degrees(math.atan2(dy, dx))
                
                # Yumuşatma: hızlı EMA + kısa pencere; gecikmeyi düşürür
                face_smoothing.append(angle)
                alpha_fast = 0.6
                if face_angle_ema is None:
                    face_angle_ema = angle
                else:
                    face_angle_ema = alpha_fast * angle + (1 - alpha_fast) * face_angle_ema
                smooth_angle = 0.5 * face_angle_ema + 0.5 * (sum(face_smoothing) / len(face_smoothing))
                
                # Daha normal hassasiyet: ~±10° eşik, yön tersini düzelt
                deadzone = 8
                if smooth_angle > deadzone:
                    move_dir = 6  # sağa eğim → sağa hareket
                elif smooth_angle < -deadzone:
                    move_dir = -6 # sola eğim → sola hareket
                else:
                    move_dir = 0
                
                # Göz kapama algılama (ateş etme)
                # Sol göz
                left_eye_top = landmarks[159]
                left_eye_bottom = landmarks[145]
                left_eye_left = landmarks[33]
                left_eye_right = landmarks[133]
                
                # Sağ göz
                right_eye_top = landmarks[386]
                right_eye_bottom = landmarks[374]
                right_eye_left = landmarks[362]
                right_eye_right = landmarks[263]
                
                # Göz açıklığı hesaplama
                left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
                left_eye_width = abs(left_eye_right.x - left_eye_left.x)
                left_eye_ratio = left_eye_height / (left_eye_width + 0.001)
                
                right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
                right_eye_width = abs(right_eye_right.x - right_eye_left.x)
                right_eye_ratio = right_eye_height / (right_eye_width + 0.001)
                
                # Facemesh'ten doğru EAR hesapla (dikey/ yatay oran)
                try:
                    left_ear = compute_ear_from_facemesh(landmarks, left=True)
                    right_ear = compute_ear_from_facemesh(landmarks, left=False)
                except Exception:
                    left_ear, right_ear = 0.0, 0.0
                avg_eye_ratio = (left_ear + right_ear) / 2
                eye_smoothing.append(avg_eye_ratio)
                smooth_eye_ratio = sum(eye_smoothing) / len(eye_smoothing)
                
                # Baseline kalibrasyonu (ilk saniyeler)
                if ear_baseline is None and len(eye_smoothing) >= 6:
                    ear_baseline = sum(eye_smoothing) / len(eye_smoothing)
                # Eşik: baseline yoksa 0.32, varsa baseline*0.70 (bir tık daha zor)
                dynamic_threshold = 0.32 if ear_baseline is None else max(0.20, min(0.45, ear_baseline * 0.70))
                ear_threshold = dynamic_threshold
                
                # Göz kapalı olduğu sürece belirli aralıkla ateş et (sürekli ateş)
                # Tek göz kırpmasına da izin ver (yanlış pozitifi azaltmak için eşik düşük zaten)
                is_closed = (left_ear < ear_threshold) or (right_ear < ear_threshold)
                if is_closed and (current_time - last_shoot_time > fire_debounce_ms):
                    shoot = True
                    last_shoot_time = current_time
        
        # Kamera görüntüsüne UI ekle
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Wave: {wave}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Health: {player.health}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Tilt head: Move | Squint/Close: Shoot", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Göz durumu göstergesi
        current_ear = (sum(eye_smoothing)/len(eye_smoothing)) if len(eye_smoothing)>0 else 0
        eye_status = "CLOSED" if len(eye_smoothing) > 0 and current_ear < ear_threshold else "OPEN"
        eye_color = (0, 0, 255) if eye_status == "CLOSED" else (0, 255, 0)
        cv2.putText(frame, f"Eyes: {eye_status}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)
        cv2.putText(frame, f"EAR: {current_ear:.2f} TH: {ear_threshold:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        
        if SHOW_CAMERA_WINDOW:
            cv2.imshow(CAM_WIN, frame)
    
    # --- Pygame Olayları ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    # Önce tam ekranı yerel çözünürlükte açmayı dene
                    try:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        info = pygame.display.Info()
                        WIDTH, HEIGHT = info.current_w, info.current_h
                    except pygame.error:
                        # Alternatifler
                        opened = False
                        for flags in (pygame.FULLSCREEN, pygame.FULLSCREEN | pygame.SCALED, pygame.NOFRAME):
                            try:
                                screen = pygame.display.set_mode((0, 0), flags)
                                info = pygame.display.Info()
                                WIDTH, HEIGHT = info.current_w, info.current_h
                                opened = True
                                break
                            except pygame.error:
                                continue
                        if not opened:
                            is_fullscreen = False
                            WIDTH, HEIGHT = SMALL_W, SMALL_H
                            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                else:
                    WIDTH, HEIGHT = SMALL_W, SMALL_H
                    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            elif event.key == pygame.K_r and game_state == 'game_over':
                # Oyunu yeniden başlat
                restart_game()
        elif event.type == pygame.VIDEORESIZE and not is_fullscreen:
            # Pencere yeniden boyutlandırıldıysa, yeni boyutlara geç
            WIDTH, HEIGHT = max(400, event.w), max(300, event.h)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            # Yıldız alanını yeni boyuta göre yeniden oluştur (mevcut sayıyı koru)
            try:
                stars = create_starfield(len(stars))
            except Exception:
                stars = create_starfield(150)
                # Oyunu yeniden başlat
                restart_game()
    
    if game_state == 'playing':
        game_over_time = None
        # --- Oyuncu Hareketi (Fizik ile) ---
        if not hasattr(player, 'velocity_x'):
            player.velocity_x = 0
        if not hasattr(player, 'acceleration'):
            player.acceleration = 0.6
        if not hasattr(player, 'friction'):
            player.friction = 0.88
        
        # Fizik hesaplaması
        # Hareketi bir tık daha hızlandır ve tepkisel yap
        player.acceleration = 0.6
        player.friction = 0.89
        player.velocity_x += move_dir * player.acceleration
        player.velocity_x *= player.friction
        
        # Maksimum hız sınırı
        max_speed = 9
        player.velocity_x = max(-max_speed, min(max_speed, player.velocity_x))
        
        # Pozisyon güncellemesi
        player.rect.x += player.velocity_x
        player.rect.x = max(0, min(WIDTH - player.rect.width, player.rect.x))
        
        # --- Ateş Etme ---
        if shoot:
            if player.weapon_type == 'triple' and player.weapon_timer > 0:
                bullets.append(Bullet(player.rect.centerx - 5, player.rect.y, -1))
                bullets.append(Bullet(player.rect.centerx - 15, player.rect.y, -1))
                bullets.append(Bullet(player.rect.centerx + 5, player.rect.y, -1))
            else:
                bullets.append(Bullet(player.rect.centerx - 4, player.rect.y, -1))
        
        # --- Düşman Oluşturma ---
        if boss is None:
            if current_time - last_enemy_spawn > max(500, 2000 - wave * 100):
                spawn_enemy()
                last_enemy_spawn = current_time
            
            # Boss oluşturma
            if enemies_killed >= wave * 10:
                boss = Boss()
                enemies.clear()
                game_state = 'boss_fight'
        
        # --- Güncellemeler ---
        player.update()
        
        # Mermiler
        bullets = [bullet for bullet in bullets if bullet.update()]
        enemy_bullets = [bullet for bullet in enemy_bullets if bullet.update()]
        
        # Düşmanlar
        new_enemies = []
        for enemy in enemies:
            if enemy.update():
                new_enemies.append(enemy)
                # Düşman ateşi
                if enemy.shoot_timer <= 0 and random.random() < 0.02:
                    enemy_bullets.append(Bullet(enemy.rect.centerx, enemy.rect.bottom, 1))
                    enemy.shoot_timer = random.randint(60, 120)
            else:
                # Düşman ekranı geçti - hasar verme (kapatıldı)
                pass
        enemies = new_enemies
        
        # Power-ups
        powerups = [powerup for powerup in powerups if powerup.update()]
        
        # Parçacıklar
        particles = [p for p in particles if p.update()]
        
        # Yıldızlar
        update_starfield(stars)
        
        # --- Çarpışmalar ---
        # Oyuncu mermileri vs düşmanlar
        for bullet in bullets[:]:
            for enemy in enemies[:]:
                if bullet.rect.colliderect(enemy.rect):
                    bullets.remove(bullet)
                    enemy.health -= 1
                    create_explosion(enemy.rect.centerx, enemy.rect.centery, 'small')
                    
                    if enemy.health <= 0:
                        enemies.remove(enemy)
                        enemies_killed += 1
                        # Düşman tipine göre skor
                        enemy_scores = {'basic': 10, 'fast': 15, 'tank': 25, 'zigzag': 20}
                        score += enemy_scores.get(enemy.type, 10)
                        spawn_powerup(enemy.rect.centerx, enemy.rect.centery)
                        create_explosion(enemy.rect.centerx, enemy.rect.centery, 'normal')
                    break
        
        # Düşman mermileri vs oyuncu
        for bullet in enemy_bullets[:]:
            if bullet.rect.colliderect(player.rect):
                enemy_bullets.remove(bullet)
                if player.take_damage(15):
                    game_state = 'game_over'
        
        # Oyuncu vs düşmanlar
        for enemy in enemies[:]:
            if enemy.rect.colliderect(player.rect):
                enemies.remove(enemy)
                if player.take_damage(25):
                    game_state = 'game_over'
                create_explosion(player.rect.centerx, player.rect.centery, 'large')
        
        # Power-ups
        for powerup in powerups[:]:
            if powerup.rect.colliderect(player.rect):
                powerups.remove(powerup)
                if powerup.type == 'triple':
                    player.weapon_type = 'triple'
                    player.weapon_timer = 600
                elif powerup.type == 'rapid':
                    player.weapon_type = 'rapid'
                    player.weapon_timer = 600
                elif powerup.type == 'shield':
                    player.shield = min(100, player.shield + 50)
                elif powerup.type == 'health':
                    player.health = min(player.max_health, player.health + 30)
    
    elif game_state == 'boss_fight':
        if boss:
            # Oyuncu hareketi (boss savaşında da aktif)
            if not hasattr(player, 'velocity_x'):
                player.velocity_x = 0
            if not hasattr(player, 'acceleration'):
                player.acceleration = 0.6
            if not hasattr(player, 'friction'):
                player.friction = 0.88
            player.velocity_x += move_dir * player.acceleration
            player.velocity_x *= player.friction
            player.rect.x += player.velocity_x
            player.rect.x = max(0, min(WIDTH - player.rect.width, player.rect.x))

            # Ateş (boss savaşında da aktif)
            if shoot:
                if player.weapon_type == 'triple' and player.weapon_timer > 0:
                    bullets.append(Bullet(player.rect.centerx - 5, player.rect.y, -1))
                    bullets.append(Bullet(player.rect.centerx - 15, player.rect.y, -1))
                    bullets.append(Bullet(player.rect.centerx + 5, player.rect.y, -1))
                else:
                    bullets.append(Bullet(player.rect.centerx - 4, player.rect.y, -1))

            player.update()
            boss.update()
            
            # Boss ateşi
            if boss.shoot_timer <= 0:
                if boss.pattern == 0:  # Tek mermi
                    enemy_bullets.append(Bullet(boss.rect.centerx, boss.rect.bottom, 1))
                    boss.shoot_timer = 30
                elif boss.pattern == 1:  # Üçlü mermi
                    for i in range(3):
                        enemy_bullets.append(Bullet(boss.rect.centerx + i*20 - 20, boss.rect.bottom, 1))
                    boss.shoot_timer = 45
                elif boss.pattern == 2:  # Yelpaze
                    for i in range(5):
                        enemy_bullets.append(Bullet(boss.rect.centerx + i*15 - 30, boss.rect.bottom, 1))
                    boss.shoot_timer = 60
            
            # Mermiler
            bullets = [bullet for bullet in bullets if bullet.update()]
            enemy_bullets = [bullet for bullet in enemy_bullets if bullet.update()]
            particles = [p for p in particles if p.update()]
            update_starfield(stars)
            
            # Boss vs mermiler
            do_restart = False
            for bullet in bullets[:]:
                if bullet.rect.colliderect(boss.rect):
                    bullets.remove(bullet)
                    boss.health -= 1
                    create_explosion(bullet.rect.centerx, bullet.rect.centery, 'small')
                    
                    if boss.health <= 0:
                        explode_x, explode_y = boss.rect.centerx, boss.rect.centery
                        create_explosion(explode_x, explode_y, 'large')
                        # Boss öldüğünde aynı frame içinde kalan kodlara
                        # girmemek için bayrak koy ve döngüden çık
                        do_restart = True
                        break
            if do_restart:
                restart_game()
                continue
            
            # Düşman mermileri vs oyuncu
            for bullet in enemy_bullets[:]:
                if bullet.rect.colliderect(player.rect):
                    enemy_bullets.remove(bullet)
                    if player.take_damage(20):
                        game_state = 'game_over'
    
    # --- Çizim ---
    screen.fill(BLACK)
    
    # Yıldızlar
    draw_starfield(screen, stars)
    
    # Oyun nesneleri
    if game_state in ['playing', 'boss_fight']:
        for bullet in bullets:
            bullet.draw(screen)
        for bullet in enemy_bullets:
            bullet.draw(screen)
        for enemy in enemies:
            enemy.draw(screen)
        for powerup in powerups:
            powerup.draw(screen)
        
        if boss:
            boss.draw(screen)
        
        player.draw(screen)
    
    # Parçacıklar
    for particle in particles:
        particle.draw(screen)
    
    # --- HUD ---
    score_text = font.render(f"Score: {score}", True, WHITE)
    wave_text = font.render(f"Wave: {wave}", True, WHITE)
    health_text = font.render(f"Health: {player.health}/{player.max_health}", True, WHITE)
    
    screen.blit(score_text, (10, 10))
    screen.blit(wave_text, (10, 50))
    screen.blit(health_text, (10, 90))
    
    if player.shield > 0:
        shield_text = font.render(f"Shield: {player.shield}", True, CYAN)
        screen.blit(shield_text, (10, 130))
    
    if player.weapon_timer > 0:
        weapon_text = font.render(f"Weapon: {player.weapon_type.upper()} ({player.weapon_timer//60}s)", True, YELLOW)
        screen.blit(weapon_text, (10, 170))
    
    # Game Over ekranı + Sonsuz döngü (auto-restart)
    if game_state == 'game_over':
        if 'game_over_time' not in globals() or game_over_time is None:
            game_over_time = pygame.time.get_ticks()
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))
        
        game_over_text = big_font.render("GAME OVER", True, RED)
        final_score_text = font.render(f"Final Score: {score}", True, WHITE)
        restarting_text = font.render("Restarting...", True, WHITE)
        
        screen.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//2 - 100))
        screen.blit(final_score_text, (WIDTH//2 - final_score_text.get_width()//2, HEIGHT//2 - 30))
        screen.blit(restarting_text, (WIDTH//2 - restarting_text.get_width()//2, HEIGHT//2 + 20))
        
        # 2 saniye sonra otomatik yeniden başlat
        if pygame.time.get_ticks() - game_over_time > 2000:
            player = Player()
            bullets.clear()
            enemy_bullets.clear()
            enemies.clear()
            powerups.clear()
            particles.clear()
            boss = None
            score = 0
            wave = 1
            enemies_killed = 0
            game_state = 'playing'
            # Blink kalibrasyonunu sıfırla
            ear_baseline = None
            eye_smoothing.clear()
            game_over_time = None
    
    # F11 bilgisi
    f11_text = font.render("F11: Fullscreen", True, (128, 128, 128))
    screen.blit(f11_text, (WIDTH - f11_text.get_width() - 10, HEIGHT - 30))
    
    pygame.display.flip()
    
    # ESC ile çıkış
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
# Mediapipe kaynaklarını kapat
try:
    face_mesh.close()
except Exception:
    pass
pygame.quit()
