import cv2
import numpy as np
import pygame
import sys
from collections import deque
import math
import random

# --- Pygame Ayarları ---
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Renk Takibi ile Breakout")
is_fullscreen = False

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)
big_font = pygame.font.SysFont(None, 48)

# Renkler
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
PURPLE = (140, 80, 180)
CYAN = (80, 200, 220)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)

# Paddle (oyuncu çubuğu)
paddle_width, paddle_height = 100, 15
paddle = pygame.Rect(WIDTH//2 - paddle_width//2, HEIGHT-40, paddle_width, paddle_height)
paddle_smoothed_x = float(paddle.centerx)
paddle_smoothing_alpha = 0.2  # 0-1 arası, küçükse daha yumuşak
recent_cx = deque(maxlen=5)
deadband_px = 6
alpha_fast = 0.25
alpha_slow = 0.08

# Top
ball = pygame.Rect(WIDTH//2, HEIGHT//2, 15, 15)
ball_speed = [4.0, -4.0]
ball_min_speed = 3.0
ball_max_speed = 10.0

def clamp_ball_speed():
    speed = math.hypot(ball_speed[0], ball_speed[1])
    if speed == 0:
        ball_speed[0], ball_speed[1] = 0.0, -ball_min_speed
        return
    # Hızı üst sınıra göre ölçekle
    if speed > ball_max_speed:
        scale = ball_max_speed / speed
        ball_speed[0] *= scale
        ball_speed[1] *= scale
    # Minimum hızı koru (duranlaşmayı önle)
    speed = math.hypot(ball_speed[0], ball_speed[1])
    if speed < ball_min_speed:
        scale = ball_min_speed / (speed if speed != 0 else 1)
        ball_speed[0] *= scale
        ball_speed[1] *= scale

score = 0

# Parçacık sistemi
class Particle:
    def __init__(self, x, y, color, velocity, life):
        self.x = x
        self.y = y
        self.color = color
        self.vx, self.vy = velocity
        self.life = life
        self.max_life = life
        self.size = random.randint(2, 5)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        self.vy *= 0.98
        self.life -= 1
        self.size = max(0, self.size * 0.95)
        return self.life > 0
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size))

particles = []

# Animasyonlu tuğla sistemi
class AnimatedBrick:
    def __init__(self, rect, color):
        self.rect = rect
        self.color = color
        self.scale = 1.0
        self.destroying = False
        self.destroy_timer = 0
    
    def start_destroy(self):
        self.destroying = True
        self.destroy_timer = 10
    
    def update(self):
        if self.destroying:
            self.destroy_timer -= 1
            self.scale = max(0, self.scale - 0.1)
            return self.destroy_timer > 0
        return True
    
    def draw(self, screen):
        if self.scale > 0:
            scaled_rect = self.rect.copy()
            scaled_rect.width = int(self.rect.width * self.scale)
            scaled_rect.height = int(self.rect.height * self.scale)
            scaled_rect.center = self.rect.center
            
            # Gölge efekti
            shadow_rect = scaled_rect.copy()
            shadow_rect.x += 2
            shadow_rect.y += 2
            pygame.draw.rect(screen, (50, 50, 50), shadow_rect)
            
            # Ana tuğla
            pygame.draw.rect(screen, self.color, scaled_rect)
            pygame.draw.rect(screen, (255, 255, 255), scaled_rect, 1)
            
            # Parlaklık efekti
            if self.scale > 0.5:
                highlight = scaled_rect.copy()
                highlight.height = int(highlight.height * 0.3)
                pygame.draw.rect(screen, (255, 255, 255), highlight)

# Tuğlalar
brick_rows, brick_cols = 5, 8
brick_width = WIDTH // brick_cols
brick_height = 25
def create_bricks():
    bricks_local = []
    palette = [ (255, 99, 71), (255, 165, 0), (255, 215, 0), (50, 205, 50), (65, 105, 225) ]
    for row in range(brick_rows):
        for col in range(brick_cols):
            brick_rect = pygame.Rect(col*brick_width, row*brick_height+50, brick_width-2, brick_height-2)
            color = palette[row % len(palette)]
            animated_brick = AnimatedBrick(brick_rect, color)
            bricks_local.append(animated_brick)
    return bricks_local
bricks = create_bricks()

def create_gradient_surface(width, height, top_color, bottom_color):
    surface = pygame.Surface((width, height))
    for y in range(height):
        t = y / max(height-1, 1)
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (width, y))
    return surface

def create_starfield(width, height, num_stars=50):
    surface = pygame.Surface((width, height))
    surface.set_colorkey((0, 0, 0))
    for _ in range(num_stars):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(1, 3)
        brightness = random.randint(100, 255)
        color = (brightness, brightness, brightness)
        pygame.draw.circle(surface, color, (x, y), size)
    return surface

bg_surface = create_gradient_surface(WIDTH, HEIGHT, (5, 10, 25), (20, 40, 80))
starfield = create_starfield(WIDTH, HEIGHT, 30)

# Top izi efekti
ball_trail = deque(maxlen=8)

def reset_game():
    global score, bricks, ball_speed, paddle_smoothed_x, recent_cx, particles
    score = 0
    bricks = create_bricks()
    ball.x = WIDTH // 2
    ball.y = HEIGHT // 2
    ball_speed[0] = 4.0
    ball_speed[1] = -4.0
    paddle.centerx = WIDTH // 2
    paddle_smoothed_x = float(paddle.centerx)
    recent_cx.clear()
    particles.clear()
    ball_trail.clear()

# --- OpenCV Ayarları ---
cap = cv2.VideoCapture(0)

# Kırmızı renk aralığı (HSV) - iki bant
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([179, 255, 255])
kernel = np.ones((3, 3), np.uint8)
min_contour_area = 600

while True:
    # OpenCV - Kamera okuma
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Gürültü azaltma (hafifçe)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            x, y, w, h = cv2.boundingRect(c)
        else:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
        if area >= min_contour_area:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        recent_cx.append(cx)
        median_cx = int(np.median(list(recent_cx)))
        target_x = float(median_cx)
        dx = target_x - paddle_smoothed_x
        alpha = alpha_slow if abs(dx) < deadband_px else alpha_fast
        paddle_smoothed_x = (1 - alpha) * paddle_smoothed_x + alpha * target_x
        paddle.centerx = max(paddle_width // 2, min(WIDTH - paddle_width // 2, int(paddle_smoothed_x)))

    cv2.imshow("Kamera", frame)

    # --- Pygame Oyun Döngüsü ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            is_fullscreen = not is_fullscreen
            try:
                flags = (pygame.FULLSCREEN | pygame.SCALED) if is_fullscreen else 0
                screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
            except pygame.error:
                try:
                    flags = pygame.FULLSCREEN if is_fullscreen else 0
                    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
                except pygame.error:
                    is_fullscreen = False
                    screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.flip()

    # Top hareketi
    ball.x += int(ball_speed[0])
    ball.y += int(ball_speed[1])

    if ball.left <= 0 or ball.right >= WIDTH:
        ball_speed[0] = -ball_speed[0]
        clamp_ball_speed()
    if ball.top <= 0:
        ball_speed[1] = -ball_speed[1]
        clamp_ball_speed()
    if ball.bottom >= HEIGHT:
        # Skoru sıfırla ve oyunu yeniden başlat
        reset_game()

    # Paddle çarpışma
    if ball.colliderect(paddle):
        # Açısal sekme
        paddle_center = paddle.centerx
        relative = (ball.centerx - paddle_center) / (paddle_width / 2)
        relative = max(-1.0, min(1.0, relative))
        max_angle = math.radians(60)
        angle = relative * max_angle
        speed_mag = max(ball_min_speed, min(ball_max_speed, math.hypot(ball_speed[0], ball_speed[1]) * 1.03))
        ball_speed[0] = speed_mag * math.sin(angle)
        ball_speed[1] = -abs(speed_mag * math.cos(angle))
        clamp_ball_speed()

    # Tuğla çarpışma
    hit_index = -1
    for i, brick in enumerate(bricks):
        if ball.colliderect(brick.rect):
            hit_index = i
            break
    
    if hit_index != -1:
        hit_brick = bricks[hit_index]
        # Parçacık efekti oluştur
        for _ in range(8):
            particle = Particle(
                hit_brick.rect.centerx, hit_brick.rect.centery,
                hit_brick.color,
                (random.uniform(-3, 3), random.uniform(-3, 3)),
                random.randint(20, 40)
            )
            particles.append(particle)
        
        hit_brick.start_destroy()
        score += 10
        ball_speed[1] = -ball_speed[1]
        clamp_ball_speed()
    
    # Animasyonlu tuğlaları güncelle
    bricks = [brick for brick in bricks if brick.update()]
    
    # Tüm tuğlalar kırılırsa yeniden oluştur ve hızı biraz arttır
    if not bricks:
        bricks = create_bricks()
        ball_speed[0] *= 1.05
        ball_speed[1] *= 1.05
        clamp_ball_speed()

    # Top izi güncelle
    ball_trail.append((ball.centerx, ball.centery))
    
    # Parçacıkları güncelle
    particles = [p for p in particles if p.update()]
    
    # Çizimler
    screen.blit(bg_surface, (0, 0))
    screen.blit(starfield, (0, 0))
    
    # Top izi çiz
    for i, (tx, ty) in enumerate(ball_trail):
        alpha = int(255 * (i / len(ball_trail)))
        size = int(15 * (i / len(ball_trail)))
        if size > 0:
            pygame.draw.circle(screen, (255, 255, 255, alpha), (tx, ty), size)
    
    # Tuğlaları çiz
    for brick in bricks:
        brick.draw(screen)
    
    # Parçacıkları çiz
    for particle in particles:
        particle.draw(screen)
    
    # Paddle gölge efekti
    shadow_paddle = paddle.copy()
    shadow_paddle.x += 3
    shadow_paddle.y += 3
    pygame.draw.rect(screen, (0, 0, 0, 100), shadow_paddle)
    
    # Ana paddle
    pygame.draw.rect(screen, (30, 160, 220), paddle)
    pygame.draw.rect(screen, (200, 240, 255), paddle, 2)
    
    # Top gölge efekti
    shadow_ball = ball.copy()
    shadow_ball.x += 2
    shadow_ball.y += 2
    pygame.draw.ellipse(screen, (0, 0, 0, 100), shadow_ball)
    
    # Ana top
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.ellipse(screen, (200, 200, 255), ball, 2)

    # HUD
    hud = font.render(f"Skor: {score}", True, (230, 230, 240))
    screen.blit(hud, (10, 10))
    
    # Skor artışı efekti
    if score > 0 and score % 50 == 0:
        combo_text = big_font.render("COMBO!", True, GOLD)
        screen.blit(combo_text, (WIDTH//2 - 80, HEIGHT//2 - 50))

    pygame.display.flip()
    clock.tick(60)

    # ESC ile çıkış
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
