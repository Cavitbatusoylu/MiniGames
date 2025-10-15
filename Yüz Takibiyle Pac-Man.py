import cv2
import mediapipe as mp
import pygame
import random
import sys
import math
import os

# --- Pygame Ayarları ---
pygame.init()
pygame.mixer.init()

WIDTH, HEIGHT = 800, 600
TILE_SIZE = 20
GRID_WIDTH = WIDTH // TILE_SIZE
GRID_HEIGHT = HEIGHT // TILE_SIZE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🎮 Yüz Takibiyle Pac-Man - Gelişmiş Versiyon")

clock = pygame.time.Clock()
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 32)
font_small = pygame.font.Font(None, 24)

# Renkler
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PINK = (255, 192, 203)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_BLUE = (0, 0, 139)

# Oyun durumu
class GameState:
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2
    PAUSED = 3

game_state = GameState.MENU
score = 0
high_score = 0
level = 1
lives = 3

# Labirent haritası (1=duvar, 0=boş, 2=yem, 3=güçlendirici)
maze = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,2,1],
    [1,3,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,3,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1],
    [1,2,2,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2,1],
    [1,1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1],
    [0,0,0,0,0,1,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,1,0,0,0,0,0],
    [0,0,0,0,0,1,2,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,2,1,0,0,0,0,0],
    [1,1,1,1,1,1,2,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,2,1,1,1,1,1,1],
    [0,0,0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,2,0,0,0,0,0,0],
    [1,1,1,1,1,1,2,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,2,1,1,1,1,1,1],
    [0,0,0,0,0,1,2,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,2,1,0,0,0,0,0],
    [0,0,0,0,0,1,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,1,0,0,0,0,0],
    [1,1,1,1,1,1,2,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,2,1,1,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,2,1],
    [1,2,2,2,1,1,2,2,2,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2,1,1,2,2,2,2,1],
    [1,1,1,2,1,1,2,1,1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,1,1,1,1,2,1,1,2,1,1,1,1],
    [1,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,1],
    [1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]

# Pac-Man sınıfı
class PacMan:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = (0, 0)
        self.next_direction = (0, 0)
        self.speed = 2
        self.radius = TILE_SIZE // 2 - 2
        self.mouth_angle = 0
        self.mouth_speed = 0.3
        
    def update(self):
        # Yön değiştirme
        if self.can_move(self.next_direction):
            self.direction = self.next_direction
            
        # Hareket
        if self.can_move(self.direction):
            self.x += self.direction[0] * self.speed
            self.y += self.direction[1] * self.speed
            
        # Ağız animasyonu
        self.mouth_angle += self.mouth_speed
        
    def can_move(self, direction):
        if direction == (0, 0):
            return True
            
        new_x = self.x + direction[0] * self.speed
        new_y = self.y + direction[1] * self.speed
        
        # Pac-Man'in dört köşesini kontrol et
        corners = [
            (new_x, new_y),  # Sol üst
            (new_x + TILE_SIZE - 1, new_y),  # Sağ üst
            (new_x, new_y + TILE_SIZE - 1),  # Sol alt
            (new_x + TILE_SIZE - 1, new_y + TILE_SIZE - 1)  # Sağ alt
        ]
        
        for corner_x, corner_y in corners:
            # Grid pozisyonu
            grid_x = int(corner_x // TILE_SIZE)
            grid_y = int(corner_y // TILE_SIZE)
            
            # Sınır kontrolü
            if grid_x < 0 or grid_x >= GRID_WIDTH or grid_y < 0 or grid_y >= GRID_HEIGHT:
                return False
                
            # Duvar kontrolü
            if maze[grid_y][grid_x] == 1:
                return False
                
        return True
        
    def draw(self, screen):
        center_x = int(self.x + TILE_SIZE // 2)
        center_y = int(self.y + TILE_SIZE // 2)
        
        # Ağız açısı hesaplama
        mouth_start = math.radians(self.mouth_angle)
        mouth_end = math.radians(360 - self.mouth_angle)
        
        # Pac-Man çizimi (sarı daire + ağız)
        pygame.draw.circle(screen, YELLOW, (center_x, center_y), self.radius)
        
        # Ağız çizimi
        if self.direction != (0, 0):
            # Yön doğrultusunda ağız
            if self.direction[0] > 0:  # Sağa
                mouth_start = math.radians(0)
                mouth_end = math.radians(360 - self.mouth_angle * 2)
            elif self.direction[0] < 0:  # Sola
                mouth_start = math.radians(180)
                mouth_end = math.radians(180 + self.mouth_angle * 2)
            elif self.direction[1] > 0:  # Aşağı
                mouth_start = math.radians(90)
                mouth_end = math.radians(90 + self.mouth_angle * 2)
            elif self.direction[1] < 0:  # Yukarı
                mouth_start = math.radians(270)
                mouth_end = math.radians(270 + self.mouth_angle * 2)
        
        # Ağız için üçgen çizimi
        if mouth_end - mouth_start < math.radians(360):
            points = [(center_x, center_y)]
            for angle in [mouth_start, mouth_end]:
                x = center_x + self.radius * math.cos(angle)
                y = center_y + self.radius * math.sin(angle)
                points.append((x, y))
            pygame.draw.polygon(screen, BLACK, points)

# Ghost sınıfı
class Ghost:
    def __init__(self, x, y, color, name):
        self.x = x
        self.y = y
        self.color = color
        self.name = name
        self.direction = (0, 0)
        self.speed = 1.5
        self.radius = TILE_SIZE // 2 - 2
        self.mode = "chase"  # chase, scatter, frightened
        self.target_x = 0
        self.target_y = 0
        
    def update(self, pacman):
        # Hedef belirleme
        if self.mode == "chase":
            self.target_x = pacman.x
            self.target_y = pacman.y
        elif self.mode == "scatter":
            # Köşelere git
            corners = [(0, 0), (GRID_WIDTH-1, 0), (0, GRID_HEIGHT-1), (GRID_WIDTH-1, GRID_HEIGHT-1)]
            corner = corners[hash(self.name) % 4]
            self.target_x = corner[0] * TILE_SIZE
            self.target_y = corner[1] * TILE_SIZE
            
        # Yön hesaplama (basit AI)
        self.calculate_direction()
        
        # Hareket
        if self.can_move(self.direction):
            self.x += self.direction[0] * self.speed
            self.y += self.direction[1] * self.speed
            
    def calculate_direction(self):
        # Pac-Man'e en kısa yolu bul
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        
        # Rastgele yön seçimi (basit AI)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        valid_directions = [d for d in directions if self.can_move(d)]
        
        if valid_directions:
            # Hedefe en yakın yönü seç
            best_direction = valid_directions[0]
            best_distance = float('inf')
            
            for direction in valid_directions:
                new_x = self.x + direction[0] * self.speed
                new_y = self.y + direction[1] * self.speed
                distance = math.sqrt((self.target_x - new_x)**2 + (self.target_y - new_y)**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_direction = direction
                    
            self.direction = best_direction
            
    def can_move(self, direction):
        if direction == (0, 0):
            return True
            
        new_x = self.x + direction[0] * self.speed
        new_y = self.y + direction[1] * self.speed
        
        # Ghost'un dört köşesini kontrol et
        corners = [
            (new_x, new_y),  # Sol üst
            (new_x + TILE_SIZE - 1, new_y),  # Sağ üst
            (new_x, new_y + TILE_SIZE - 1),  # Sol alt
            (new_x + TILE_SIZE - 1, new_y + TILE_SIZE - 1)  # Sağ alt
        ]
        
        for corner_x, corner_y in corners:
            # Grid pozisyonu
            grid_x = int(corner_x // TILE_SIZE)
            grid_y = int(corner_y // TILE_SIZE)
            
            # Sınır kontrolü
            if grid_x < 0 or grid_x >= GRID_WIDTH or grid_y < 0 or grid_y >= GRID_HEIGHT:
                return False
                
            # Duvar kontrolü
            if maze[grid_y][grid_x] == 1:
                return False
                
        return True
        
    def draw(self, screen):
        center_x = int(self.x + TILE_SIZE // 2)
        center_y = int(self.y + TILE_SIZE // 2)
        
        # Ghost gövdesi
        pygame.draw.circle(screen, self.color, (center_x, center_y), self.radius)
        
        # Ghost alt kısmı (dalgalı)
        points = []
        for i in range(5):
            x = center_x - self.radius + (i * self.radius * 2 // 4)
            y = center_y + self.radius - (10 if i % 2 == 0 else 0)
            points.append((x, y))
        points.append((center_x + self.radius, center_y + self.radius))
        points.append((center_x - self.radius, center_y + self.radius))
        pygame.draw.polygon(screen, self.color, points)
        
        # Gözler
        eye_size = 3
        pygame.draw.circle(screen, WHITE, (center_x - 4, center_y - 3), eye_size)
        pygame.draw.circle(screen, WHITE, (center_x + 4, center_y - 3), eye_size)
        pygame.draw.circle(screen, BLACK, (center_x - 4, center_y - 3), 1)
        pygame.draw.circle(screen, BLACK, (center_x + 4, center_y - 3), 1)

# Oyun nesneleri
pacman = PacMan(20 * TILE_SIZE, 20 * TILE_SIZE)
ghosts = [
    Ghost(19 * TILE_SIZE, 9 * TILE_SIZE, RED, "Blinky"),
    Ghost(20 * TILE_SIZE, 9 * TILE_SIZE, PINK, "Pinky"),
    Ghost(21 * TILE_SIZE, 9 * TILE_SIZE, CYAN, "Inky"),
    Ghost(22 * TILE_SIZE, 9 * TILE_SIZE, ORANGE, "Clyde")
]

# Yüz takibi için değişkenler
nose_sx, nose_sy = None, None
NOSE_ALPHA = 0.6
power_mode = False
power_timer = 0

# Mediapipe ayarları
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

def draw_maze():
    for y in range(len(maze)):
        for x in range(len(maze[y])):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            
            if maze[y][x] == 1:  # Duvar
                pygame.draw.rect(screen, BLUE, rect)
                pygame.draw.rect(screen, DARK_BLUE, rect, 1)
            elif maze[y][x] == 2:  # Yem
                center_x = x * TILE_SIZE + TILE_SIZE // 2
                center_y = y * TILE_SIZE + TILE_SIZE // 2
                pygame.draw.circle(screen, WHITE, (center_x, center_y), 2)
            elif maze[y][x] == 3:  # Güçlendirici
                center_x = x * TILE_SIZE + TILE_SIZE // 2
                center_y = y * TILE_SIZE + TILE_SIZE // 2
                pygame.draw.circle(screen, WHITE, (center_x, center_y), 6)

def check_collisions():
    global score, lives, power_mode, power_timer
    
    # Yem yeme
    pacman_grid_x = int(pacman.x // TILE_SIZE)
    pacman_grid_y = int(pacman.y // TILE_SIZE)
    
    if 0 <= pacman_grid_x < GRID_WIDTH and 0 <= pacman_grid_y < GRID_HEIGHT:
        if maze[pacman_grid_y][pacman_grid_x] == 2:  # Normal yem
            maze[pacman_grid_y][pacman_grid_x] = 0
            score += 10
        elif maze[pacman_grid_y][pacman_grid_x] == 3:  # Güçlendirici
            maze[pacman_grid_y][pacman_grid_x] = 0
            score += 50
            power_mode = True
            power_timer = 300  # 10 saniye (30 FPS)
    
    # Ghost çarpışması
    pacman_rect = pygame.Rect(pacman.x, pacman.y, TILE_SIZE, TILE_SIZE)
    
    for ghost in ghosts:
        ghost_rect = pygame.Rect(ghost.x, ghost.y, TILE_SIZE, TILE_SIZE)
        
        if pacman_rect.colliderect(ghost_rect):
            if power_mode:
                # Ghost'u yeme
                ghost.x = 20 * TILE_SIZE
                ghost.y = 9 * TILE_SIZE
                score += 200
            else:
                # Ölüm
                lives -= 1
                if lives <= 0:
                    return "game_over"
                else:
                    # Yeniden başlat
                    pacman.x = 20 * TILE_SIZE
                    pacman.y = 20 * TILE_SIZE
                    pacman.direction = (0, 0)
                    pacman.next_direction = (0, 0)
    
    return "playing"

def draw_ui():
    # Skor
    score_text = font_medium.render(f"Skor: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    
    # Canlar
    lives_text = font_medium.render(f"Can: {lives}", True, WHITE)
    screen.blit(lives_text, (10, 40))
    
    # Seviye
    level_text = font_medium.render(f"Seviye: {level}", True, WHITE)
    screen.blit(level_text, (10, 70))
    
    # Güçlendirici modu
    if power_mode:
        power_text = font_small.render("GÜÇLÜ!", True, YELLOW)
        screen.blit(power_text, (WIDTH - 100, 10))

def draw_menu():
    screen.fill(BLACK)
    
    title = font_large.render("🎮 YÜZ TAKİBİYLE PAC-MAN", True, YELLOW)
    title_rect = title.get_rect(center=(WIDTH//2, HEIGHT//2 - 100))
    screen.blit(title, title_rect)
    
    subtitle = font_medium.render("Yüzünüzü hareket ettirerek Pac-Man'i kontrol edin!", True, WHITE)
    subtitle_rect = subtitle.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
    screen.blit(subtitle, subtitle_rect)
    
    start_text = font_medium.render("SPACE - Başla", True, WHITE)
    start_rect = start_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 50))
    screen.blit(start_text, start_rect)
    
    high_score_text = font_small.render(f"En Yüksek Skor: {high_score}", True, WHITE)
    high_score_rect = high_score_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 100))
    screen.blit(high_score_text, high_score_rect)

def draw_game_over():
    screen.fill(BLACK)
    
    game_over_text = font_large.render("OYUN BİTTİ!", True, RED)
    game_over_rect = game_over_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
    screen.blit(game_over_text, game_over_rect)
    
    final_score_text = font_medium.render(f"Final Skor: {score}", True, WHITE)
    final_score_rect = final_score_text.get_rect(center=(WIDTH//2, HEIGHT//2))
    screen.blit(final_score_text, final_score_rect)
    
    restart_text = font_medium.render("SPACE - Tekrar Oyna", True, WHITE)
    restart_rect = restart_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 50))
    screen.blit(restart_text, restart_rect)

def reset_game():
    global score, lives, level, power_mode, power_timer
    global pacman, ghosts, maze
    
    score = 0
    lives = 3
    level = 1
    power_mode = False
    power_timer = 0
    
    # Pac-Man pozisyonu
    pacman.x = 20 * TILE_SIZE
    pacman.y = 20 * TILE_SIZE
    pacman.direction = (0, 0)
    pacman.next_direction = (0, 0)
    
    # Ghost pozisyonları
    ghosts[0].x = 19 * TILE_SIZE
    ghosts[0].y = 9 * TILE_SIZE
    ghosts[1].x = 20 * TILE_SIZE
    ghosts[1].y = 9 * TILE_SIZE
    ghosts[2].x = 21 * TILE_SIZE
    ghosts[2].y = 9 * TILE_SIZE
    ghosts[3].x = 22 * TILE_SIZE
    ghosts[3].y = 9 * TILE_SIZE
    
    # Labirenti sıfırla
    for y in range(len(maze)):
        for x in range(len(maze[y])):
            if maze[y][x] == 0:
                maze[y][x] = 2  # Yemleri geri getir

# Ana oyun döngüsü
running = True
while running:
    # Kamera okuma
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Yüz takibi
    move = (0, 0)
    if results.multi_face_landmarks and game_state == GameState.PLAYING:
        for face in results.multi_face_landmarks:
            nose = face.landmark[1]
            nx, ny = nose.x * WIDTH, nose.y * HEIGHT
            
            if nose_sx is None:
                nose_sx, nose_sy = nx, ny
            else:
                nose_sx = NOSE_ALPHA * nx + (1 - NOSE_ALPHA) * nose_sx
                nose_sy = NOSE_ALPHA * ny + (1 - NOSE_ALPHA) * nose_sy
                
            x, y = int(nose_sx), int(nose_sy)
            cx, cy = WIDTH // 2, HEIGHT // 2
            dx, dy = x - cx, y - cy
            thrx = max(30, int(WIDTH * 0.08))
            thry = max(30, int(HEIGHT * 0.08))

            if abs(dx) > abs(dy):
                if dx < -thrx:
                    move = (-1, 0)  # Sol
                elif dx > thrx:
                    move = (1, 0)   # Sağ
            else:
                if dy < -thry:
                    move = (0, -1)  # Yukarı
                elif dy > thry:
                    move = (0, 1)   # Aşağı

    # Pygame olayları
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if game_state == GameState.MENU:
                    game_state = GameState.PLAYING
                    reset_game()
                elif game_state == GameState.GAME_OVER:
                    game_state = GameState.MENU
                    if score > high_score:
                        high_score = score
            elif event.key == pygame.K_ESCAPE:
                running = False

    # Oyun güncellemeleri
    if game_state == GameState.PLAYING:
        # Pac-Man kontrolü
        if move != (0, 0):
            pacman.next_direction = move
            
        pacman.update()
        
        # Ghost güncellemeleri
        for ghost in ghosts:
            ghost.update(pacman)
            
        # Çarpışma kontrolü
        result = check_collisions()
        if result == "game_over":
            game_state = GameState.GAME_OVER
            
        # Güçlendirici modu
        if power_mode:
            power_timer -= 1
            if power_timer <= 0:
                power_mode = False
                
        # Seviye tamamlama kontrolü
        all_dots_eaten = True
        for row in maze:
            if 2 in row or 3 in row:
                all_dots_eaten = False
                break
                
        if all_dots_eaten:
            level += 1
            reset_game()

    # Çizim
    screen.fill(BLACK)
    
    if game_state == GameState.MENU:
        draw_menu()
    elif game_state == GameState.PLAYING:
        draw_maze()
        pacman.draw(screen)
        for ghost in ghosts:
            ghost.draw(screen)
        draw_ui()
    elif game_state == GameState.GAME_OVER:
        draw_game_over()

    pygame.display.flip()
    clock.tick(30)

    # Kamera gösterimi
    cv2.imshow("Yüz Takibi - Pac-Man", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()