import cv2
import pygame
import numpy as np
import mediapipe as mp
import random
import time

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 256, 640
BLOCK_SIZE = 32
screen = pygame.display.set_mode((WIDTH + 150, HEIGHT))  # extra space for next piece & score
pygame.display.set_caption("Eye-Controlled Tetris")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
colors = [
    (0, 255, 255),  # Cyan for I
    (255, 255, 0),  # Yellow for O
    (128, 0, 128),  # Purple for T
    (255, 165, 0),  # Orange for L
    (0, 0, 255),    # Blue for J
    (0, 255, 0),    # Green for S
    (255, 0, 0),    # Red for Z
]

# Tetris grid dimensions
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE

# Shapes
shapes = [
    [[1, 1, 1, 1]],        # I
    [[1, 1], [1, 1]],      # O
    [[0, 1, 0], [1, 1, 1]], # T
    [[1, 0, 0], [1, 1, 1]], # L
    [[0, 0, 1], [1, 1, 1]], # J
    [[1, 1, 0], [0, 1, 1]], # S
    [[0, 1, 1], [1, 1, 0]], # Z
]
shape_colors = colors

class EyeTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        
    def get_gaze_position(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, frame
        frame = cv2.flip(frame, 1)
        cam_h, cam_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        gaze_x = None

        # Use nose position for head control if detected.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                gaze_x = nose.x  # normalized coordinate between 0 and 1

                # Draw markers for eyes and nose
                left_eye = face_landmarks.landmark[159]
                right_eye = face_landmarks.landmark[386]
                left_eye_px = (int(left_eye.x * cam_w), int(left_eye.y * cam_h))
                right_eye_px = (int(right_eye.x * cam_w), int(right_eye.y * cam_h))
                nose_px = (int(nose.x * cam_w), int(nose.y * cam_h))
                cv2.circle(frame, left_eye_px, 5, (0, 0, 255), -1)
                cv2.circle(frame, right_eye_px, 5, (0, 0, 255), -1)
                cv2.circle(frame, nose_px, 5, (255, 0, 0), -1)

                # Calculate and show blink ratio (for reference)
                left_eye_ratio = (face_landmarks.landmark[159].y - face_landmarks.landmark[145].y) / (
                                    face_landmarks.landmark[33].x - face_landmarks.landmark[133].x)
                right_eye_ratio = (face_landmarks.landmark[386].y - face_landmarks.landmark[374].y) / (
                                    face_landmarks.landmark[362].x - face_landmarks.landmark[263].x)
                blink_ratio = (left_eye_ratio + right_eye_ratio) / 2
                cv2.putText(frame, f"Blink:{blink_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)
                break  # use first detected face

        # Fall back to mouse control if no face detected
        if gaze_x is None:
            mouse_x = pygame.mouse.get_pos()[0]
            gaze_x = mouse_x / float(WIDTH)

        # Overlay grid and gaze marker on camera frame
        gaze_pixel = int(gaze_x * cam_w)
        cv2.line(frame, (gaze_pixel, 0), (gaze_pixel, cam_h), (0, 255, 0), 2)
        for i in range(1, 4):
            cv2.line(frame, (int(i * cam_w / 4), 0), (int(i * cam_w / 4), cam_h), (255, 255, 255), 1)
            cv2.line(frame, (0, int(i * cam_h / 4)), (cam_w, int(i * cam_h / 4)), (255, 255, 255), 1)
        return gaze_x, frame

    def is_blink(self):
        keys = pygame.key.get_pressed()
        return keys[pygame.K_SPACE]

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

class Piece:
    def __init__(self, shape):
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.x = GRID_WIDTH // 2 - len(shape[0]) // 2
        self.y = 0

    def rotate(self):
        rotated = [list(row) for row in zip(*self.shape[::-1])]
        return rotated

def attempt_rotation(piece, grid):
    new_shape = piece.rotate()
    if valid_space(new_shape, grid, (piece.x, piece.y)):
        return new_shape
    # Try small offsets left and right (wall kick)
    for dx in [-1, 1]:
        if valid_space(new_shape, grid, (piece.x + dx, piece.y)):
            piece.x += dx
            return new_shape
    return piece.shape

def create_grid(locked_positions={}):
    grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if (x, y) in locked_positions:
                grid[y][x] = locked_positions[(x, y)]
    return grid

def valid_space(shape, grid, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                if x + off_x < 0 or x + off_x >= GRID_WIDTH or y + off_y >= GRID_HEIGHT:
                    return False
                if grid[y + off_y][x + off_x] != BLACK:
                    return False
    return True

def clear_rows(grid, locked_positions):
    lines_cleared = 0
    for y in range(GRID_HEIGHT - 1, -1, -1):
        if BLACK not in grid[y]:
            lines_cleared += 1
            for x in range(GRID_WIDTH):
                try:
                    del locked_positions[(x, y)]
                except KeyError:
                    continue
            for key in sorted(list(locked_positions), key=lambda k: k[1], reverse=True):
                x, y_pos = key
                if y_pos < y:
                    locked_positions[(x, y_pos + 1)] = locked_positions.pop(key)
    return lines_cleared

def draw_grid(surface, grid):
    surface.fill(BLACK)
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            pygame.draw.rect(surface, grid[y][x],
                             (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
    for x in range(GRID_WIDTH):
        pygame.draw.line(surface, WHITE, (x * BLOCK_SIZE, 0),
                         (x * BLOCK_SIZE, HEIGHT))
    for y in range(GRID_HEIGHT):
        pygame.draw.line(surface, WHITE, (0, y * BLOCK_SIZE),
                         (WIDTH, y * BLOCK_SIZE))

def draw_next_piece(surface, piece):
    font = pygame.font.SysFont('Arial', 20)
    label = font.render('Next Piece:', True, WHITE)
    surface.blit(label, (WIDTH + 10, 10))
    for y, row in enumerate(piece.shape):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, piece.color,
                                 (WIDTH + 10 + x * BLOCK_SIZE, 40 + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

def draw_score(surface, score, level, lines):
    font = pygame.font.SysFont('Arial', 20)
    score_label = font.render(f"Score: {score}", True, WHITE)
    level_label = font.render(f"Level: {level}", True, WHITE)
    lines_label = font.render(f"Lines: {lines}", True, WHITE)
    surface.blit(score_label, (WIDTH + 10, 150))
    surface.blit(level_label, (WIDTH + 10, 180))
    surface.blit(lines_label, (WIDTH + 10, 210))

def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont('Arial', size)
    label = font.render(text, True, color)
    surface.blit(label, (WIDTH // 2 - label.get_width() // 2,
                          HEIGHT // 2 - label.get_height() // 2))

def draw_piece(surface, piece):
    for y, row in enumerate(piece.shape):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, piece.color,
                                 ((piece.x + x) * BLOCK_SIZE,
                                  (piece.y + y) * BLOCK_SIZE,
                                  BLOCK_SIZE, BLOCK_SIZE), 0)

def main():
    clock = pygame.time.Clock()
    locked_positions = {}
    grid = create_grid(locked_positions)
    current_piece = Piece(random.choice(shapes))
    next_piece = Piece(random.choice(shapes))
    eye_tracker = EyeTracker()
    fall_time = 0
    fall_speed = 500  # milliseconds per block fall
    score = 0
    level = 1
    lines_cleared_total = 0
    paused = False
    game_over = False
    running = True
    prev_blink = False  # for rising-edge blink detection

    while running:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r and game_over:
                    game_over = False
                    locked_positions = {}
                    grid = create_grid(locked_positions)
                    current_piece = Piece(random.choice(shapes))
                    next_piece = Piece(random.choice(shapes))
                    score = 0
                    level = 1
                    lines_cleared_total = 0
                    fall_speed = 500

            if event.type == pygame.MOUSEBUTTONDOWN:
                new_shape = attempt_rotation(current_piece, grid)
                current_piece.shape = new_shape

        if paused:
            screen.fill(BLACK)
            draw_text_middle(screen, "PAUSED", 40, WHITE)
            pygame.display.update()
            continue

        # --- Head (gaze) control ---
        gaze_x, cam_frame = eye_tracker.get_gaze_position()
        if gaze_x is not None:
            target_x = int(gaze_x * GRID_WIDTH)
            if valid_space(current_piece.shape, grid, (target_x, current_piece.y)):
                current_piece.x = target_x

        # --- Rotate on blink using rising edge of spacebar ---
        blink_now = eye_tracker.is_blink()
        if blink_now and not prev_blink:
            new_shape = attempt_rotation(current_piece, grid)
            current_piece.shape = new_shape
        prev_blink = blink_now

        # Piece falling
        if fall_time > fall_speed:
            fall_time = 0
            if valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y + 1)):
                current_piece.y += 1
            else:
                for y, row in enumerate(current_piece.shape):
                    for x, cell in enumerate(row):
                        if cell:
                            locked_positions[(current_piece.x + x, current_piece.y + y)] = current_piece.color
                cleared = clear_rows(grid, locked_positions)
                if cleared > 0:
                    score += cleared * 10
                    lines_cleared_total += cleared
                    level = lines_cleared_total // 10 + 1
                    fall_speed = max(100, 500 - (level - 1) * 50)
                current_piece = next_piece
                next_piece = Piece(random.choice(shapes))
                if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                    game_over = True

        # Game Over screen
        if game_over:
            while game_over:
                screen.fill(BLACK)
                draw_text_middle(screen, "GAME OVER", 40, WHITE)
                draw_text_middle(screen, "Press R to Restart", 20, WHITE)
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_over = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        game_over = False
                        locked_positions = {}
                        grid = create_grid(locked_positions)
                        current_piece = Piece(random.choice(shapes))
                        next_piece = Piece(random.choice(shapes))
                        score = 0
                        level = 1
                        lines_cleared_total = 0
                        fall_speed = 500
                clock.tick(5)
            continue

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if (x, y) in locked_positions:
                    grid[y][x] = locked_positions[(x, y)]
        draw_grid(screen, grid)
        draw_next_piece(screen, next_piece)
        draw_score(screen, score, level, lines_cleared_total)
        draw_piece(screen, current_piece)
        pygame.display.update()

        cv2.imshow("Camera Feed", cam_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False

    eye_tracker.release()
    pygame.quit()

if __name__ == "__main__":
    main()