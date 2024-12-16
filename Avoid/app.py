import sys
import cv2
import mediapipe as mp
import pygame
import random

# ---------------------------
# Mediapipe Hands Setup
# ---------------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# Game Constants
# ---------------------------
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400

PLAYER_X = 100
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 40

# Vertical levels
NORMAL_LEVEL = 250
JUMP_LEVEL = 150
DUCK_LEVEL = 350
MOVE_SPEED = 5

OBSTACLE_WIDTH = 40
OBSTACLE_HEIGHT = 40
OBSTACLE_SPEED = 6

FPS = 30

def reset_game():
    return {
        'player_y': NORMAL_LEVEL,
        'score': 0,
        'game_over': False,
        'start': False,
        'obstacles': []
    }

game_state = reset_game()

# ---------------------------
# Pygame Initialization
# ---------------------------
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Finger-Controlled Flying Runner")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

def draw_text(surface, text, x, y, color=(0,0,0)):
    img = font.render(text, True, color)
    rect = img.get_rect()
    rect.center = (x, y)
    surface.blit(img, rect)

# ---------------------------
# Video Capture
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open video capture.")
    sys.exit()

running = True
while running:
    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read camera frame
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw landmarks on camera frame
    annotated_frame = frame.copy()
    finger_detected = False
    target_y = NORMAL_LEVEL  # Default if no finger detected

    if results.multi_hand_landmarks:
        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Index finger tip is landmark 8
        index_tip = hand_landmarks.landmark[8]
        finger_x = int(index_tip.x * width)
        finger_y = int(index_tip.y * height)
        finger_detected = True

        # Draw a red dot on the index fingertip
        cv2.circle(annotated_frame, (finger_x, finger_y), 10, (0,0,255), -1)

        # Determine vertical zone
        # Divide the frame into three zones:
        # Top third: y < height/3
        # Middle third: height/3 <= y <= 2*height/3
        # Bottom third: y > 2*height/3
        top_zone = height / 3
        middle_zone = 2 * height / 3

        if finger_y < top_zone:
            # High finger => jump
            target_y = JUMP_LEVEL
            print("Finger detected: JUMP zone")
        elif finger_y < middle_zone:
            # Middle => normal
            target_y = NORMAL_LEVEL
            print("Finger detected: NORMAL zone")
        else:
            # Low => duck
            target_y = DUCK_LEVEL
            print("Finger detected: DUCK zone")
    else:
        # No hand detected, maintain normal level
        target_y = NORMAL_LEVEL
        print("No finger detected, normal stance")

    cv2.imshow("Camera Feed (Hand)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit from camera
        running = False

    # START SCREEN
    if not game_state['start']:
        screen.fill((255,255,255))
        draw_text(screen, "Finger-Controlled Flying Runner", WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 50)
        draw_text(screen, "Press 'S' to start", WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        draw_text(screen, "Press 'Q' to quit", WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50)
        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_s]:
            game_state['start'] = True
        if keys[pygame.K_q]:
            running = False

        clock.tick(FPS)
        continue

    # GAME OVER SCREEN
    if game_state['game_over']:
        screen.fill((255,255,255))
        draw_text(screen, "GAME OVER", WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 50, (255,0,0))
        draw_text(screen, f"Score: {game_state['score']}", WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        draw_text(screen, "Press 'R' to Restart or 'Q' to Quit", WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50)
        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            game_state = reset_game()
        if keys[pygame.K_q]:
            running = False

        clock.tick(FPS)
        continue

    # MAIN GAME LOOP
    current_y = game_state['player_y']
    # Move towards target_y smoothly
    if abs(current_y - target_y) > MOVE_SPEED:
        if current_y < target_y:
            current_y += MOVE_SPEED
        else:
            current_y -= MOVE_SPEED
    else:
        current_y = target_y

    game_state['player_y'] = current_y

    # Spawn obstacles at random vertical positions
    if random.randint(0, 100) < 5:
        obs_y = random.randint(150, 350)
        game_state['obstacles'].append([WINDOW_WIDTH, obs_y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT])

    # Move obstacles
    for obs in game_state['obstacles']:
        obs[0] -= OBSTACLE_SPEED

    # Remove off-screen obstacles
    game_state['obstacles'] = [o for o in game_state['obstacles'] if o[0] + o[2] > 0]

    # Check collisions
    player_rect = pygame.Rect(PLAYER_X, game_state['player_y'], PLAYER_WIDTH, PLAYER_HEIGHT)
    for obs in game_state['obstacles']:
        obs_rect = pygame.Rect(obs[0], obs[1], obs[2], obs[3])
        if player_rect.colliderect(obs_rect):
            game_state['game_over'] = True

    # Increase score
    game_state['score'] += 1

    # Render Game
    screen.fill((255,255,255))

    # Draw player
    pygame.draw.rect(screen, (0,128,0), (PLAYER_X, game_state['player_y'], PLAYER_WIDTH, PLAYER_HEIGHT))

    # Draw obstacles
    for obs in game_state['obstacles']:
        pygame.draw.rect(screen, (128,0,0), obs)

    # Draw score
    draw_text(screen, f"Score: {game_state['score']}", WINDOW_WIDTH - 100, 50)

    pygame.display.flip()
    clock.tick(FPS)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
