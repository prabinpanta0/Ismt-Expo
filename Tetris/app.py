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
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[0, 1, 0], [1, 1, 1]],  # T
    [[1, 0, 0], [1, 1, 1]],  # L
    [[0, 0, 1], [1, 1, 1]],  # J
    [[1, 1, 0], [0, 1, 1]],  # S
    [[0, 1, 1], [1, 1, 0]],  # Z
]
shape_colors = colors

class EyeTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.prev_eyes_detected = True
        self.blink_detected = False
        self.blink_start_time = None

    def get_gaze_position(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        gaze_x = None
        eyes_detected = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_ratio = (face_landmarks.landmark[159].y - face_landmarks.landmark[145].y) / (face_landmarks.landmark[33].x - face_landmarks.landmark[133].x)
                right_eye_ratio = (face_landmarks.landmark[386].y - face_landmarks.landmark[374].y) / (face_landmarks.landmark[362].x - face_landmarks.landmark[263].x)
                blink_ratio = (left_eye_ratio + right_eye_ratio) / 2
                eyes_detected = blink_ratio >= 0.2
        # For simplicity, we simulate gaze_x from the mouse position when available
        mouse_x = pygame.mouse.get_pos()[0]
        gaze_x = mouse_x / float(WIDTH)
        return gaze_x, frame

    def is_blink(self):
        # For demo purpose, simulate blink with spacebar press
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
    for y in range(GRID_HEIGHT-1, -1, -1):
        if BLACK not in grid[y]:
            lines_cleared += 1
            for x in range(GRID_WIDTH):
                try:
                    del locked_positions[(x, y)]
                except:
                    continue
            # Shift rows downward
            for key in sorted(list(locked_positions), key=lambda k: k[1], reverse=True):
                x, y_pos = key
                if y_pos < y:
                    locked_positions[(x, y_pos + 1)] = locked_positions.pop(key)
    return lines_cleared

def draw_grid(surface, grid):
    surface.fill(BLACK)
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            pygame.draw.rect(surface, grid[y][x], (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
    for x in range(GRID_WIDTH):
        pygame.draw.line(surface, WHITE, (x * BLOCK_SIZE, 0), (x * BLOCK_SIZE, HEIGHT))
    for y in range(GRID_HEIGHT):
        pygame.draw.line(surface, WHITE, (0, y * BLOCK_SIZE), (WIDTH, y * BLOCK_SIZE))

def draw_next_piece(surface, piece):
    font = pygame.font.SysFont('Arial', 20)
    label = font.render('Next Piece:', True, WHITE)
    surface.blit(label, (WIDTH + 10, 10))
    for y, row in enumerate(piece.shape):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, piece.color, (WIDTH + 10 + x * BLOCK_SIZE, 40 + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

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
    surface.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - label.get_height() // 2))

def draw_piece(surface, piece):
    for y, row in enumerate(piece.shape):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, piece.color,
                                 ((piece.x + x) * BLOCK_SIZE, (piece.y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

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
    
    running = True
    game_over = False

    while running:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Game Over screen
        if game_over:
            screen.fill(BLACK)
            draw_text_middle(screen, "GAME OVER", 40, WHITE)
            pygame.display.update()
            pygame.time.delay(1500)
            # Reset game variables
            locked_positions = {}
            grid = create_grid(locked_positions)
            current_piece = Piece(random.choice(shapes))
            next_piece = Piece(random.choice(shapes))
            score = 0
            level = 1
            lines_cleared_total = 0
            fall_speed = 500
            game_over = False

        # Eye control input (simulate via mouse)
        gaze_x, _ = eye_tracker.get_gaze_position()
        if gaze_x is not None:
            target_x = int(gaze_x * GRID_WIDTH)
            if valid_space(current_piece.shape, grid, (target_x, current_piece.y)):
                current_piece.x = target_x

        # Check for blink to rotate piece
        if eye_tracker.is_blink():
            new_shape = current_piece.rotate()
            # Only update if rotation is valid
            if valid_space(new_shape, grid, (current_piece.x, current_piece.y)):
                current_piece.shape = new_shape

        # Piece falling
        if fall_time > fall_speed:
            fall_time = 0
            if valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y + 1)):
                current_piece.y += 1
            else:
                # Lock piece
                for y, row in enumerate(current_piece.shape):
                    for x, cell in enumerate(row):
                        if cell:
                            locked_positions[(current_piece.x + x, current_piece.y + y)] = current_piece.color
                # Clear rows and update score/level
                cleared = clear_rows(grid, locked_positions)
                if cleared > 0:
                    score += cleared * 10
                    lines_cleared_total += cleared
                    level = lines_cleared_total // 10 + 1
                    fall_speed = max(100, 500 - (level - 1) * 50)
                # Spawn next piece
                current_piece = next_piece
                next_piece = Piece(random.choice(shapes))
                # Check game over
                if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                    game_over = True

        # Draw locked pieces into grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if (x, y) in locked_positions:
                    grid[y][x] = locked_positions[(x, y)]
        draw_grid(screen, grid)
        draw_next_piece(screen, next_piece)
        draw_score(screen, score, level, lines_cleared_total)
        draw_piece(screen, current_piece)  # Draw the falling piece
        pygame.display.update()

    eye_tracker.release()
    pygame.quit()

if __name__ == "__main__":
    main()