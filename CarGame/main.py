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

def get_head_tilt_angle(image):
    """ Detect head tilt angle using mediapipe face landmarks. 
        Returns roll angle in degrees. Positive tilt = right tilt, negative = left tilt.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        # Using left outer eye corner (33) and right outer eye corner (263)
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]

        x1, y1 = left_eye.x, left_eye.y
        x2, y2 = right_eye.x, right_eye.y

        dy = y2 - y1
        dx = x2 - x1
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees
    return 0.0

# -------------- PYGAME SETUP ------------------
pygame.init()

# Screen dimensions
WIDTH = 480
HEIGHT = 640
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3-Lane Head Tilt Controlled Game")

clock = pygame.time.Clock()

# -------------- GAME VARIABLES ------------------
# Define road and lane dimensions
road_width = WIDTH * 0.6
road_x = (WIDTH - road_width) / 2
lane_width = road_width / 3

# Lane centers
lanes = [
    road_x + lane_width * 0.5,
    road_x + lane_width * 1.5,
    road_x + lane_width * 2.5
]

current_lane_index = 1  # start in the center lane

# Load car image
car_image = pygame.image.load("car.png").convert_alpha()

# Dynamic scaling according to lane width:
# Let's say the car width should be about 50% of lane width
car_scale_factor = 0.5
desired_car_width = int(lane_width * car_scale_factor)
original_aspect_ratio = car_image.get_height() / car_image.get_width()
desired_car_height = int(desired_car_width * original_aspect_ratio)
car_image = pygame.transform.scale(car_image, (desired_car_width, desired_car_height))

car_width = desired_car_width
car_height = desired_car_height

car_x = lanes[current_lane_index] - car_width // 2
car_y = HEIGHT - car_height - 20

# Load obstacle image
obstacle_image = pygame.image.load("bus.png").convert_alpha()
obstacle_image = pygame.transform.scale(obstacle_image, (55, 100))

coin_color = (255, 223, 0)
bg_color = (34, 139, 34)

score = 0
game_over = False

obstacles = []
coins = []
spawn_timer = 0
spawn_interval = 60  # spawn interval

# Movement thresholds
tilt_threshold = 10  # degrees of tilt needed to move lanes

# Speed of scrolling
scroll_speed = 10

# To prevent multiple lane changes in a single tilt event:
allowed_to_change_lane = True

# Track time for increasing difficulty
start_time = time.time()

def spawn_objects():
    # Randomly spawn obstacles or coins in one of the three lanes
    lane = random.randint(0, 2)
    x = lanes[lane] - 20
    y = -40
    # 50% chance obstacle, 50% coin
    if random.random() < 0.5:
        obstacles.append(pygame.Rect(x, y, 40, 40))
    else:
        coins.append(pygame.Rect(x, y, 40, 40))

def move_car(direction):
    global current_lane_index
    if direction == "left" and current_lane_index > 0:
        current_lane_index -= 1
    elif direction == "right" and current_lane_index < 2:
        current_lane_index += 1

running = True
while running:
    # --- EVENT HANDLING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if game_over:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Restart the game
                    score = 0
                    game_over = False
                    obstacles = []
                    coins = []
                    current_lane_index = 1  # Reset to center lane
                    car_x = lanes[current_lane_index] - car_width // 2
                    allowed_to_change_lane = True
                    start_time = time.time()  # Reset start time
                    scroll_speed = 10  # Reset speed
                    spawn_interval = 60  # Reset spawn interval
                elif event.key == pygame.K_q:
                    running = False

    if game_over:
        # Draw game over screen
        screen.fill(bg_color)
        font = pygame.font.SysFont(None, 36)
        game_over_surf = font.render("GAME OVER!", True, (255, 0, 0))
        prompt_surf = font.render(f"Score: {score}", True, (255, 255, 255))
        restart_surf = font.render("Press R to Restart or Q to Quit", True, (255, 255, 255))
        screen.blit(game_over_surf, (WIDTH//2 - game_over_surf.get_width()//2, HEIGHT//2 - 60))
        screen.blit(prompt_surf, (WIDTH//2 - prompt_surf.get_width()//2, HEIGHT//2 - 20))
        screen.blit(restart_surf, (WIDTH//2 - restart_surf.get_width()//2, HEIGHT//2 + 20))
        pygame.display.flip()
        clock.tick(30)
        continue  # Skip updating game logic when game over

    # --- HEAD TILT DETECTION ---
    ret, frame = cap.read()
    if not ret:
        print("Camera not found.")
        running = False
        continue

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    angle = get_head_tilt_angle(frame)

    # Decide if we move the car:
    if allowed_to_change_lane:
        if angle < -tilt_threshold:
            move_car("left")
            allowed_to_change_lane = False
        elif angle > tilt_threshold:
            move_car("right")
            allowed_to_change_lane = False

    # Reset allowed_to_change_lane if angle returns to neutral zone
    if -tilt_threshold <= angle <= tilt_threshold:
        allowed_to_change_lane = True

    # Update car position based on lane:
    car_x = lanes[current_lane_index] - car_width // 2

    # --- SPAWN OBJECTS ---
    spawn_timer += 1
    if spawn_timer >= spawn_interval:
        spawn_timer = 0
        spawn_objects()

    # --- UPDATE OBJECTS ---
    for obs in obstacles:
        obs.y += scroll_speed
    for c in coins:
        c.y += scroll_speed

    # Remove off-screen objects
    obstacles = [o for o in obstacles if o.y < HEIGHT]
    coins = [c for c in coins if c.y < HEIGHT]

    # Check collisions
    car_rect = pygame.Rect(car_x, car_y, car_width, car_height)

    # Collision with obstacles -> game over
    for o in obstacles:
        if car_rect.colliderect(o):
            game_over = True

    # Collision with coins -> increase score and remove coin
    collected_coins = []
    for idx, c in enumerate(coins):
        if car_rect.colliderect(c):
            score += 1
            collected_coins.append(idx)
    coins = [c for i, c in enumerate(coins) if i not in collected_coins]

    # --- RENDERING ---
    screen.fill(bg_color)

    # Draw road
    pygame.draw.rect(screen, (100, 100, 100), (road_x, 0, road_width, HEIGHT))

    # Draw lane dividers
    lane_line_color = (255, 255, 255)
    lane_1_x = road_x + road_width / 3
    lane_2_x = road_x + 2 * road_width / 3
    pygame.draw.line(screen, lane_line_color, (lane_1_x, 0), (lane_1_x, HEIGHT), 2)
    pygame.draw.line(screen, lane_line_color, (lane_2_x, 0), (lane_2_x, HEIGHT), 2)

    # Draw car
    screen.blit(car_image, (car_x, car_y))

    # Draw obstacles
    for o in obstacles:
        screen.blit(obstacle_image, (o.x, o.y))

    # Draw coins
    for c in coins:
        pygame.draw.ellipse(screen, coin_color, c)

    # Draw score
    font = pygame.font.SysFont(None, 36)
    score_surf = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_surf, (10, 10))

    pygame.display.flip()
    clock.tick(30)

    # Increase difficulty every 10 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time > 10:
        scroll_speed += 1
        spawn_interval = max(20, spawn_interval - 5)  # Decrease spawn interval but not less than 20
        start_time = time.time()  # Reset start time

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()