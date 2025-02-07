import pygame
import sys
import cv2
import mediapipe as mp
import math
import random
import time

# -------------- HEAD TILT DETECTION SETUP ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
cap = cv2.VideoCapture(0)  # Use default webcam

# Global variable to track current direction for camera overlay
current_direction = "CENTER"

def get_head_tilt_angle(image):
    """Detect head tilt angle using mediapipe face landmarks.
    Returns roll angle in degrees.
    Draws markers on the eyes, additional face landmarks (including nose tip),
    and a grid over the camera feed.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    angle_degrees = 0.0
    h, w, _ = image.shape

    # Draw grid overlay (3x3 grid)
    num_rows, num_cols = 3, 3
    for i in range(1, num_cols):
        x = int(w * i / num_cols)
        cv2.line(image, (x, 0), (x, h), (200, 200, 200), 1)
    for i in range(1, num_rows):
        y = int(h * i / num_rows)
        cv2.line(image, (0, y), (w, y), (200, 200, 200), 1)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        # Using left outer eye corner (33) and right outer eye corner (263)
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]
        left_pt = (int(left_eye.x * w), int(left_eye.y * h))
        right_pt = (int(right_eye.x * w), int(right_eye.y * h))
        # Draw markers for eyes
        cv2.circle(image, left_pt, 5, (0, 255, 0), -1)
        cv2.circle(image, right_pt, 5, (0, 255, 0), -1)
        # Calculate head tilt angle based on eyes
        dy = right_eye.y - left_eye.y
        dx = right_eye.x - left_eye.x
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        # Draw additional face landmarks (blue dots)
        for idx, landmark in enumerate(face.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        # Draw nose tip (landmark 1) as a red circle for clarity
        nose = face.landmark[1]
        nose_pt = (int(nose.x * w), int(nose.y * h))
        cv2.circle(image, nose_pt, 4, (0, 0, 255), -1)

    return angle_degrees

# -------------- PYGAME SETUP ------------------
pygame.init()

# Screen dimensions
WIDTH = 480
HEIGHT = 640
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3-Lane Head Tilt Controlled Game")

clock = pygame.time.Clock()

def draw_background():
    """Draws a sky and ground gradient background."""
    sky_color = (135, 206, 235)    # Sky blue
    ground_color = (34, 139, 34)   # Forest green
    pygame.draw.rect(screen, sky_color, (0, 0, WIDTH, int(HEIGHT * 0.6)))
    pygame.draw.rect(screen, ground_color, (0, int(HEIGHT * 0.6), WIDTH, int(HEIGHT * 0.4)))

# -------------- GAME VARIABLES ------------------
road_width = WIDTH * 0.6
road_x = (WIDTH - road_width) / 2
lane_width = road_width / 3

lanes = [
    road_x + lane_width * 0.5,
    road_x + lane_width * 1.5,
    road_x + lane_width * 2.5
]

current_lane_index = 1  # Start in the center lane

car_image = pygame.image.load("car.png").convert_alpha()
car_scale_factor = 0.5  # Car width is 50% of lane width
desired_car_width = int(lane_width * car_scale_factor)
original_aspect_ratio = car_image.get_height() / car_image.get_width()
desired_car_height = int(desired_car_width * original_aspect_ratio)
car_image = pygame.transform.scale(car_image, (desired_car_width, desired_car_height))
car_width = desired_car_width
car_height = desired_car_height
car_x = lanes[current_lane_index] - car_width // 2
car_y = HEIGHT - car_height - 20

obstacle_image = pygame.image.load("bus.png").convert_alpha()
obstacle_image = pygame.transform.scale(obstacle_image, (55, 100))

coin_color = (255, 223, 0)

score = 0
game_over = False
paused = False

obstacles = []
coins = []
spawn_timer = 0
spawn_interval = 60  # Spawn interval

tilt_threshold = 10  # Degrees needed to move lanes

scroll_speed = 10
allowed_to_change_lane = True
start_time = time.time()

def spawn_objects():
    lane = random.randint(0, 2)
    x = lanes[lane] - 20
    y = -40
    if random.random() < 0.5:
        obstacles.append(pygame.Rect(x, y, 40, 40))
    else:
        coins.append(pygame.Rect(x, y, 40, 40))

def move_car(direction):
    global current_lane_index, current_direction
    if direction == "left" and current_lane_index > 0:
        current_lane_index -= 1
        current_direction = "LEFT"
    elif direction == "right" and current_lane_index < 2:
        current_lane_index += 1
        current_direction = "RIGHT"

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            # Allow quitting from pause screen by pressing Q
            if event.key == pygame.K_q and paused:
                running = False
            if event.key == pygame.K_p and not game_over:
                paused = not paused
            if game_over:
                if event.key == pygame.K_r:
                    score = 0
                    game_over = False
                    obstacles = []
                    coins = []
                    current_lane_index = 1
                    car_x = lanes[current_lane_index] - car_width // 2
                    allowed_to_change_lane = True
                    start_time = time.time()
                    scroll_speed = 10
                    spawn_interval = 60
                elif event.key == pygame.K_q:
                    running = False

    # PAUSE STATE: Freeze game updates
    if paused:
        draw_background()
        pause_overlay = pygame.Surface((WIDTH, HEIGHT))
        pause_overlay.set_alpha(180)
        pause_overlay.fill((0, 0, 0))
        font_large = pygame.font.SysFont(None, 48)
        font_small = pygame.font.SysFont(None, 36)
        paused_surf = font_large.render("GAME PAUSED", True, (255, 255, 0))
        score_surf = font_small.render(f"Score: {score}", True, (255, 255, 255))
        instruct_surf = font_small.render("Press P to resume | Q to quit", True, (200, 200, 200))
        screen.blit(paused_surf, (WIDTH // 2 - paused_surf.get_width() // 2, HEIGHT // 2 - 80))
        screen.blit(score_surf, (WIDTH // 2 - score_surf.get_width() // 2, HEIGHT // 2 - 20))
        screen.blit(instruct_surf, (WIDTH // 2 - instruct_surf.get_width() // 2, HEIGHT // 2 + 40))
        pygame.display.flip()

        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            get_head_tilt_angle(frame)
            cv2.putText(frame, f"Direction: {current_direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Camera", frame)
            cv2.waitKey(1)
        clock.tick(60)
        continue

    # GAME OVER STATE: Freeze game updates as well
    if game_over:
        draw_background()
        # Draw road and lane dividers for appearance
        pygame.draw.rect(screen, (100, 100, 100), (road_x, 0, road_width, HEIGHT))
        lane_line_color = (255, 255, 255)
        lane_1_x = road_x + road_width / 3
        lane_2_x = road_x + 2 * road_width / 3
        pygame.draw.line(screen, lane_line_color, (lane_1_x, 0), (lane_1_x, HEIGHT), 2)
        pygame.draw.line(screen, lane_line_color, (lane_2_x, 0), (lane_2_x, HEIGHT), 2)
        # Draw car, obstacles, and coins (frozen positions)
        screen.blit(car_image, (car_x, car_y))
        for o in obstacles:
            screen.blit(obstacle_image, (o.x, o.y))
        for c in coins:
            pygame.draw.ellipse(screen, coin_color, c)
        # Overlay Game Over screen
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        font_large = pygame.font.SysFont(None, 48)
        font_small = pygame.font.SysFont(None, 36)
        game_over_surf = font_large.render("GAME OVER!", True, (255, 0, 0))
        score_go_surf = font_small.render(f"Score: {score}", True, (255, 255, 255))
        restart_surf = font_small.render("Press R to Restart or Q to Quit", True, (255, 255, 255))
        screen.blit(game_over_surf, (WIDTH // 2 - game_over_surf.get_width() // 2, HEIGHT // 2 - 80))
        screen.blit(score_go_surf, (WIDTH // 2 - score_go_surf.get_width() // 2, HEIGHT // 2 - 20))
        screen.blit(restart_surf, (WIDTH // 2 - restart_surf.get_width() // 2, HEIGHT // 2 + 40))
        pygame.display.flip()
        clock.tick(60)
        continue

    # Game is active (not paused or game over)
    ret, frame = cap.read()
    if not ret:
        print("Camera not found.")
        running = False
        continue

    frame = cv2.flip(frame, 1)
    angle = get_head_tilt_angle(frame)
    cv2.putText(frame, f"Direction: {current_direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Camera", frame)

    if allowed_to_change_lane:
        if angle < -tilt_threshold:
            move_car("left")
            allowed_to_change_lane = False
        elif angle > tilt_threshold:
            move_car("right")
            allowed_to_change_lane = False

    if -tilt_threshold <= angle <= tilt_threshold:
        allowed_to_change_lane = True

    car_x = lanes[current_lane_index] - car_width // 2

    # Only update obstacles/coins and collisions when game is active
    spawn_timer += 1
    if spawn_timer >= spawn_interval:
        spawn_timer = 0
        spawn_objects()

    for obs in obstacles:
        obs.y += scroll_speed
    for c in coins:
        c.y += scroll_speed

    obstacles = [o for o in obstacles if o.y < HEIGHT]
    coins = [c for c in coins if c.y < HEIGHT]

    car_rect = pygame.Rect(car_x, car_y, car_width, car_height)
    for o in obstacles:
        if car_rect.colliderect(o):
            game_over = True

    collected_coins = []
    for idx, c in enumerate(coins):
        if car_rect.colliderect(c):
            score += 1
            collected_coins.append(idx)
    coins = [c for i, c in enumerate(coins) if i not in collected_coins]

    # Render the frame
    draw_background()
    pygame.draw.rect(screen, (100, 100, 100), (road_x, 0, road_width, HEIGHT))
    lane_line_color = (255, 255, 255)
    lane_1_x = road_x + road_width / 3
    lane_2_x = road_x + 2 * road_width / 3
    pygame.draw.line(screen, lane_line_color, (lane_1_x, 0), (lane_1_x, HEIGHT), 2)
    pygame.draw.line(screen, lane_line_color, (lane_2_x, 0), (lane_2_x, HEIGHT), 2)
    screen.blit(car_image, (car_x, car_y))
    for o in obstacles:
        screen.blit(obstacle_image, (o.x, o.y))
    for c in coins:
        pygame.draw.ellipse(screen, coin_color, c)

    font_small = pygame.font.SysFont(None, 36)
    score_surf = font_small.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_surf, (10, 10))

    pygame.display.flip()
    clock.tick(60)

    elapsed_time = time.time() - start_time
    if elapsed_time > 10:
        scroll_speed += 1
        spawn_interval = max(20, spawn_interval - 5)
        start_time = time.time()

cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()