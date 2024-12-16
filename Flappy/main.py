import cv2
import numpy as np
import random
import pygame
import time
import mediapipe as mp
import logging

# Initialize Pygame for game components
pygame.init()

# Game Variables
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 30
PIPE_GAP = 200
PIPE_WIDTH = 80
PIPE_SPEED = 5
ACCELERATION = 0.1  # Increases speed gradually
FONT = pygame.font.SysFont("Arial", 30)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# MediaPipe Hands Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load bird image
bird_image = pygame.image.load('bird.png')
bird_image = pygame.transform.scale(bird_image, (50, 50))

# Low-Pass Filter for Smoothing
def low_pass_filter(new_value, prev_value, alpha=0.2):
    return alpha * new_value + (1 - alpha) * prev_value

# Functions
def draw_text(surface, text, color, x, y):
    img = FONT.render(text, True, color)
    surface.blit(img, (x, y))

def detect_finger(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    finger_position = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * frame.shape[1])
            y = int(index_finger_tip.y * frame.shape[0])
            finger_position = (x, y)
            # Draw marker on finger
            cv2.circle(frame, (x, y), 10, RED, -1)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            break
    return finger_position

def generate_pipe():
    pipe_height = random.randint(150, 450)
    top_pipe = pygame.Rect(WINDOW_WIDTH, 0, PIPE_WIDTH, pipe_height)
    bottom_pipe = pygame.Rect(WINDOW_WIDTH, pipe_height + PIPE_GAP, PIPE_WIDTH, WINDOW_HEIGHT - pipe_height - PIPE_GAP)
    return top_pipe, bottom_pipe

def main():
    # Game State Variables
    running = True
    paused = False
    game_over = False
    score = 0
    bird_y = WINDOW_HEIGHT // 2
    prev_bird_y = bird_y
    pipes = []
    speed = PIPE_SPEED
    pipe_interval = 2.5  # Initial interval for pipe generation

    # Start time for pipes
    last_pipe_time = time.time()
    start_time = time.time()  # Track the start time for difficulty increase

    # OpenCV Camera Setup
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Pygame Screen Setup
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Finger Controlled")
    clock = pygame.time.Clock()

    while running:
        screen.fill(BLUE)
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip camera for mirror effect
        frame = cv2.flip(frame, 1)
        finger_position = detect_finger(frame)

        # Smooth Bird Position Update
        if finger_position and not paused and not game_over:
            bird_y = low_pass_filter(int(finger_position[1] * WINDOW_HEIGHT / 480), prev_bird_y)
        prev_bird_y = bird_y

        bird_rect = pygame.Rect(100, int(bird_y), 50, 50)

        # Generate Pipes
        if not game_over and not paused and time.time() - last_pipe_time > pipe_interval:
            pipes.append(generate_pipe())
            last_pipe_time = time.time()

        # Move Pipes and Check Collision
        if not paused and not game_over:
            for top_pipe, bottom_pipe in pipes:
                top_pipe.x -= int(speed)
                bottom_pipe.x -= int(speed)
                pygame.draw.rect(screen, GREEN, top_pipe)
                pygame.draw.rect(screen, GREEN, bottom_pipe)
                if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                    game_over = True
                    logging.info("Collision detected. Game over.")

        # Remove Off-screen Pipes
        pipes = [(tp, bp) for tp, bp in pipes if tp.x + PIPE_WIDTH > 0]

        # Update Score and Difficulty
        if not game_over and not paused:
            for top_pipe, _ in pipes:
                if top_pipe.x + PIPE_WIDTH < 100 < top_pipe.x + PIPE_WIDTH + speed:
                    score += 1
                    logging.info(f"Score updated: {score}")
            speed += ACCELERATION / FPS
            global PIPE_GAP
            PIPE_GAP = max(100, PIPE_GAP - 0.001)

            # Increase pipe frequency over time
            elapsed_time = time.time() - start_time
            pipe_interval = max(1.0, 2.5 - elapsed_time / 60)  # Decrease interval but not less than 1 second

        # Draw Bird
        screen.blit(bird_image, (bird_rect.x, bird_rect.y))

        # Draw Score
        draw_text(screen, f"Score: {score}", WHITE, 10, 10)

        # Pause Screen
        if paused:
            draw_text(screen, "PAUSED - Press P to Resume", YELLOW, 200, WINDOW_HEIGHT // 2)
            draw_text(screen, f"Score: {score}", YELLOW, 200, WINDOW_HEIGHT // 2 + 50)

        # Game Over Screen
        if game_over:
            draw_text(screen, "GAME OVER! Press R to Restart or Q to Quit", RED, 100, WINDOW_HEIGHT // 2)
            draw_text(screen, f"Score: {score}", RED, 100, WINDOW_HEIGHT // 2 + 50)

        # OpenCV Display
        cv2.putText(frame, "Finger Control Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Feed", frame)

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quit Game
                    running = False
                if event.key == pygame.K_r and game_over:  # Restart Game
                    main()  # Clean Restart
                    return
                if event.key == pygame.K_p:  # Pause/Unpause Game
                    paused = not paused

        # Refresh Screen
        pygame.display.flip()
        clock.tick(FPS)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()