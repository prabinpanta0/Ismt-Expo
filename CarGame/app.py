import pygame
import sys
import cv2
import random
import os

# ----------------------------
# Configuration
# ----------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

CAR_WIDTH = 50
CAR_HEIGHT = 100

OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 100

FPS = 30

# "Insane" difficulty parameters
OBSTACLE_SPEED = 15   # Speed at which obstacles fall
SPAWN_INTERVAL = 20   # Frames between spawns

# Lane boundaries (Car can only move within these horizontal limits)
LANE_LEFT_BOUND = SCREEN_WIDTH // 4
LANE_RIGHT_BOUND = SCREEN_WIDTH * 3 // 4 - CAR_WIDTH

# Paths to Haar Cascades (adjust if needed)
face_cascade_path = 'haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascade_eye.xml'
nose_cascade_path = 'haarcascade_mcs_nose.xml'

# ----------------------------
# Setup Pygame
# ----------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Insane Car Dodge")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)
small_font = pygame.font.SysFont(None, 36)

# Load images
car_img = pygame.image.load('car.png')
car_img = pygame.transform.scale(car_img, (CAR_WIDTH, CAR_HEIGHT))

obstacle_img = pygame.image.load('obstacles.png')
obstacle_img = pygame.transform.scale(obstacle_img, (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

# ----------------------------
# OpenCV Setup for Face, Eye, Nose Detection
# ----------------------------
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
nose_cascade = cv2.CascadeClassifier(nose_cascade_path)

cap = cv2.VideoCapture(0)  # Use default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# ----------------------------
# Game Entities
# ----------------------------
class PlayerCar:
    def __init__(self):
        self.x = (LANE_LEFT_BOUND + LANE_RIGHT_BOUND) // 2
        self.y = SCREEN_HEIGHT - CAR_HEIGHT - 10
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
    
    def draw(self, surface):
        surface.blit(car_img, (self.x, self.y))

    def update_position(self, target_x):
        # Constrain to lane
        constrained_x = max(LANE_LEFT_BOUND, min(LANE_RIGHT_BOUND, target_x))
        self.x = constrained_x

class Obstacle:
    def __init__(self):
        # Spawn obstacle within the lane boundaries
        self.x = random.randint(LANE_LEFT_BOUND, LANE_RIGHT_BOUND)
        self.y = -OBSTACLE_HEIGHT
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT
    
    def update(self):
        self.y += OBSTACLE_SPEED

    def draw(self, surface):
        surface.blit(obstacle_img, (self.x, self.y))
    
    def off_screen(self):
        return self.y > SCREEN_HEIGHT

    def collide(self, player):
        # Simple AABB collision check
        return (self.x < player.x + player.width and
                self.x + self.width > player.x and
                self.y < player.y + player.height and
                self.y + self.height > player.y)

# ----------------------------
# Utility Functions
# ----------------------------
def detect_face_eyes_nose(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_center_x = None
    eye_boxes = []
    nose_boxes = []
    if len(faces) > 0:
        # Consider the first face
        (x, y, w, h) = faces[0]
        face_center_x = x + w//2
        face_roi_gray = gray[y:y+h, x:x+w]

        # Detect eyes within face region
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            eye_boxes.append((x+ex, y+ey, ew, eh))

        # Detect nose within face region
        nose = nose_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
        for (nx, ny, nw, nh) in nose:
            nose_boxes.append((x+nx, y+ny, nw, nh))
    else:
        faces = []

    return face_center_x, faces, eye_boxes, nose_boxes

def map_face_x_to_game_x(face_x, frame_width):
    # Map the face X position in camera feed to the lane width
    ratio = face_x / float(frame_width)
    lane_width = (LANE_RIGHT_BOUND - LANE_LEFT_BOUND)
    return int(LANE_LEFT_BOUND + ratio * lane_width)

def draw_text_center(surface, text, font, color, y):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=(SCREEN_WIDTH//2, y))
    surface.blit(rendered, rect)

def show_camera_frame():
    # Grab a frame and show in separate window
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        # Detect face, eyes, nose just for display
        face_x, faces, eye_boxes, nose_boxes = detect_face_eyes_nose(frame)

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)

        for (ex, ey, ew, eh) in eye_boxes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        for (nx, ny, nw, nh) in nose_boxes:
            cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

def start_menu():
    # Start menu loop
    in_menu = True
    while in_menu:
        # Show camera feed
        show_camera_frame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    in_menu = False  # Start the game

        screen.fill((0, 0, 0))
        draw_text_center(screen, "Press S to Start the Game", font, (255, 255, 255), SCREEN_HEIGHT // 2)
        pygame.display.flip()
        clock.tick(FPS)

def game_loop():
    player = PlayerCar()
    obstacles = []
    score = 0
    frame_count = 0
    prev_game_x = player.x  # To track direction

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Flip frame horizontally for a mirror-like experience
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        # Detect face, eyes, nose
        face_x, faces, eye_boxes, nose_boxes = detect_face_eyes_nose(frame)
        head_detected = (face_x is not None)

        # Update player position if head detected
        if head_detected:
            game_x = map_face_x_to_game_x(face_x, frame_width)
            player.update_position(game_x)

            # Determine direction
            if player.x < prev_game_x:
                print("Moving Left")
            elif player.x > prev_game_x:
                print("Moving Right")
            else:
                print("Centered")

            prev_game_x = player.x

        # Show camera feed with markings
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)

        for (ex, ey, ew, eh) in eye_boxes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        for (nx, ny, nw, nh) in nose_boxes:
            cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Spawn obstacles
        frame_count += 1
        if frame_count % SPAWN_INTERVAL == 0:
            obstacles.append(Obstacle())

        # Update obstacles
        for obs in obstacles:
            obs.update()

        # Remove off-screen obstacles
        obstacles = [obs for obs in obstacles if not obs.off_screen()]

        # Check collisions
        for obs in obstacles:
            if obs.collide(player):
                # Game Over
                return score  # Return score to show on game over screen

        # Increase score for survival
        score += 1

        # Draw the game world
        screen.fill((30, 30, 30))
        player.draw(screen)
        for obs in obstacles:
            obs.draw(screen)

        # Draw score
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        # Head detection status
        if head_detected:
            status_text = small_font.render("Head detected", True, (0, 255, 0))
        else:
            status_text = small_font.render("No head detected", True, (255, 0, 0))
        screen.blit(status_text, (10, 60))

        pygame.display.flip()
        clock.tick(FPS)

    return score

def game_over_screen(final_score):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Restart
                    return True
                elif event.key == pygame.K_q:
                    # Quit
                    pygame.quit()
                    cv2.destroyAllWindows()
                    sys.exit()

        screen.fill((0, 0, 0))
        draw_text_center(screen, "GAME OVER", font, (255, 0, 0), SCREEN_HEIGHT // 2 - 50)
        draw_text_center(screen, f"Your Score: {final_score}", font, (255, 255, 255), SCREEN_HEIGHT // 2)
        draw_text_center(screen, "Press R to Restart or Q to Quit", small_font, (255, 255, 255), SCREEN_HEIGHT // 2 + 50)

        pygame.display.flip()
        clock.tick(FPS)

# ----------------------------
# Main loop
# ----------------------------
if __name__ == "__main__":
    # Show start menu first
    start_menu()
    # Then start the game
    while True:
        final_score = game_loop()
        restart = game_over_screen(final_score)
        if not restart:
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()
