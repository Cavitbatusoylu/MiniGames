import cv2
import mediapipe as mp
import pygame
import random
import sys

# --- Pygame Ayarları ---
pygame.init()
WIDTH, HEIGHT = 600, 600
TILE_SIZE = 30
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Yüz Takibiyle Pac-Man")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# Pac-Man
pacman = pygame.Rect(WIDTH//2, HEIGHT//2, TILE_SIZE, TILE_SIZE)
direction = (0, 0)

# Yemler
foods = []
for _ in range(20):
    x = random.randint(0, WIDTH//TILE_SIZE-1) * TILE_SIZE
    y = random.randint(0, HEIGHT//TILE_SIZE-1) * TILE_SIZE
    foods.append(pygame.Rect(x, y, TILE_SIZE, TILE_SIZE))

score = 0

# --- Mediapipe Ayarları (Yüz takibi) ---
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# Hassas ve akıcı yön tespiti için burun ucu yumuşatma
nose_sx, nose_sy = None, None
NOSE_ALPHA = 0.6  # 0-1, büyükse daha hızlı tepki

while True:
    # Kamera okuma
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    move = (0, 0)
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            # Burun ucu landmark (id=1)
            nose = face.landmark[1]
            nx, ny = nose.x * WIDTH, nose.y * HEIGHT
            if nose_sx is None:
                nose_sx, nose_sy = nx, ny
            else:
                nose_sx = NOSE_ALPHA * nx + (1 - NOSE_ALPHA) * nose_sx
                nose_sy = NOSE_ALPHA * ny + (1 - NOSE_ALPHA) * nose_sy
            x, y = int(nose_sx), int(nose_sy)

            # Orta noktayı referans al
            cx, cy = WIDTH // 2, HEIGHT // 2
            dx, dy = x - cx, y - cy
            thrx = max(20, int(WIDTH * 0.06))   # ~%6 genişlik veya min 20px
            thry = max(20, int(HEIGHT * 0.06))  # ~%6 yükseklik veya min 20px

            if abs(dx) > abs(dy):
                if dx < -thrx:
                    move = (-1, 0)
                elif dx > thrx:
                    move = (1, 0)
            else:
                if dy < -thry:
                    move = (0, -1)
                elif dy > thry:
                    move = (0, 1)

    if move != (0,0):
        direction = move

    cv2.imshow("Kamera", frame)

    # --- Pygame Oyun Döngüsü ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()

    pacman.x += direction[0] * TILE_SIZE
    pacman.y += direction[1] * TILE_SIZE

    # Duvar sınırları
    if pacman.x < 0: pacman.x = 0
    if pacman.x > WIDTH-TILE_SIZE: pacman.x = WIDTH-TILE_SIZE
    if pacman.y < 0: pacman.y = 0
    if pacman.y > HEIGHT-TILE_SIZE: pacman.y = HEIGHT-TILE_SIZE

    # Yemleri yeme
    for food in foods[:]:
        if pacman.colliderect(food):
            foods.remove(food)
            score += 1

    # Çizimler
    screen.fill((0,0,0))
    pygame.draw.rect(screen, (255,255,0), pacman)  # Pac-Man (sarı kare)
    for food in foods:
        pygame.draw.rect(screen, (255,0,0), food)

    score_text = font.render(f"Skor: {score}", True, (255,255,255))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(30)

    # ESC ile çıkış
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
