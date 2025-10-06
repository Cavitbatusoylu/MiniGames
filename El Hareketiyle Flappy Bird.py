
import cv2
import mediapipe as mp
import pygame
import random
import sys

# --- Pygame Ayarları ---
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)  # Başlangıçta tam ekran
WIDTH, HEIGHT = screen.get_size()
pygame.display.set_caption("El Hareketiyle Flappy Bird")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

# Renkler
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
BLACK = (0, 0, 0)
RED = (200, 0, 0)

# --- Mediapipe Ayarları ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Başlangıçta tam ekran açık
is_fullscreen = True

def create_pipe():
    pipe_width = 70
    pipe_gap = 200
    y = random.randint(100, HEIGHT - 200)
    top = pygame.Rect(WIDTH, 0, pipe_width, y)
    bottom = pygame.Rect(WIDTH, y + pipe_gap, pipe_width, HEIGHT - y - pipe_gap)
    return top, bottom

def game_loop():
    global is_fullscreen, screen, WIDTH, HEIGHT

    # Kuş
    bird = pygame.Rect(100, HEIGHT // 2, 40, 40)
    gravity = 0.5
    bird_velocity = 0

    # Borular
    pipes = []
    pipes.extend(create_pipe())
    score = 0
    pipe_width = 70

    while True:
        # Kamera okuma
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        jump = False
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Orta parmak ucu (id=12) ile avuç ortası (id=0)
                y_tip = handLms.landmark[12].y
                y_palm = handLms.landmark[0].y
                if y_tip < y_palm:
                    jump = True
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Kamera", frame)

        # --- Pygame Oyun Döngüsü ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # ESC ile çıkış
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_F11:  # F11 ile tam ekran ↔ pencere
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        WIDTH, HEIGHT = screen.get_size()
                    else:
                        screen = pygame.display.set_mode((800, 600))
                        WIDTH, HEIGHT = 800, 600

        if jump:
            bird_velocity = -7

        bird_velocity += gravity
        bird.y += int(bird_velocity)

        # Borular hareketi
        for pipe in pipes:
            pipe.x -= 5

        if pipes and pipes[0].x < -pipe_width:
            pipes = pipes[2:]  # ilk iki boruyu sil
            pipes.extend(create_pipe())
            score += 1

        # Çarpışma kontrolü
        game_over = False
        for pipe in pipes:
            if bird.colliderect(pipe):
                game_over = True
        if bird.top <= 0 or bird.bottom >= HEIGHT:
            game_over = True

        if game_over:
            # Game Over ekranı
            screen.fill(WHITE)
            text1 = font.render(f"Oyun Bitti!", True, RED)
            text2 = font.render(f"Skor: {score}", True, BLACK)
            text3 = font.render("Yeniden basliyor...", True, GREEN)
            screen.blit(text1, (WIDTH // 2 - 100, HEIGHT // 2 - 60))
            screen.blit(text2, (WIDTH // 2 - 80, HEIGHT // 2))
            screen.blit(text3, (WIDTH // 2 - 150, HEIGHT // 2 + 60))
            pygame.display.flip()
            pygame.time.wait(2000)
            return  # yeniden oyun başlat

        # Çizimler
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLUE, bird)
        for pipe in pipes:
            pygame.draw.rect(screen, GREEN, pipe)

        score_text = font.render(f"Skor: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(30)

        # ESC ile çıkış (OpenCV penceresinden)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()

# --- Sonsuz Döngü ---
while True:
    game_loop()
