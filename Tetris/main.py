import cv2
import pygame
import numpy as np
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 320, 640
BLOCK_SIZE = 32
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Eye-Controlled Tetris")

# Define colors
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

# Tetris grid
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
        self.face_mesh = self.mp_face_mesh.FaceMesh()
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

                nose_point = (face_landmarks.landmark[1].x * WIDTH, face_landmarks.landmark[1].y * HEIGHT)
                eyes_detected = True
                gaze_x = nose_point[0] / WIDTH

                if self.prev_eyes_detected and not eyes_detected:
                    self.blink_start_time = pygame.time.get_ticks()
                elif not self.prev_eyes_detected and eyes_detected:
                    blink_duration = pygame.time.get_ticks() - self.blink_start_time
                    if 50 < blink_duration < 400:
                        self.blink_detected = True
                    else:
                        self.blink_detected = False
                    self.blink_start_time = None
                else:
                    self.blink_detected = False
                self.prev_eyes_detected = eyes_detected

                # Draw face and eye landmarks
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow('Eye Tracking', cv2.resize(frame, (400, 300)))
        cv2.waitKey(1)
        return gaze_x, frame

    def is_blink(self):
        return self.blink_detected

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
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

def create_grid(locked_positions={}):
    grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if (x, y) in locked_positions:
                grid[y][x] = locked_positions[(x, y)]
    return grid

def draw_grid(surface, grid):
    surface.fill(BLACK)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            pygame.draw.rect(surface, grid[y][x], (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
    for x in range(GRID_WIDTH):
        pygame.draw.line(surface, WHITE, (x * BLOCK_SIZE, 0), (x * BLOCK_SIZE, HEIGHT))
    for y in range(GRID_HEIGHT):
        pygame.draw.line(surface, WHITE, (0, y * BLOCK_SIZE), (WIDTH, y * BLOCK_SIZE))

def clear_rows(grid, locked_positions):
    lines_cleared = 0
    for y in range(len(grid)-1, -1, -1):
        row = grid[y]
        if BLACK not in row:
            lines_cleared += 1
            for x in range(GRID_WIDTH):
                del locked_positions[(x, y)]
            for pos in sorted(locked_positions, key=lambda k: k[1])[::-1]:
                x, y_pos = pos
                if y_pos < y:
                    new_key = (x, y_pos + 1)
                    locked_positions[new_key] = locked_positions.pop(pos)
    return lines_cleared

def draw_text_middle(surface, text, size, color, score=None):
    font = pygame.font.SysFont('Arial', size)
    label = font.render(text, True, color)
    surface.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - label.get_height() // 2))
    if score is not None:
        score_label = font.render(f"Score: {score}", True, color)
        surface.blit(score_label, (WIDTH // 2 - score_label.get_width() // 2, HEIGHT // 2 + label.get_height()))

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

def check_game_over(locked_positions):
    for x in range(GRID_WIDTH):
        if (x, 0) in locked_positions:
            return True
    return False

def draw_next_piece(surface, piece):
    font = pygame.font.SysFont('Arial', 20)
    label = font.render('Next Piece', True, WHITE)
    surface.blit(label, (WIDTH + 10, 10))
    for y, row in enumerate(piece.shape):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, piece.color, (WIDTH + 10 + x * BLOCK_SIZE, 30 + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

def main():
    eye_tracker = EyeTracker()
    locked_positions = {}
    grid = create_grid(locked_positions)
    current_piece = Piece(shapes[np.random.randint(0, len(shapes))])
    next_piece = Piece(shapes[np.random.randint(0, len(shapes))])
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.5
    score = 0
    level = 1
    lines_cleared = 0

    running = True
    game_over = False
    paused = False
    prev_gaze_x = None
    while running:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_p:
                    paused = not paused

        if game_over:
            draw_text_middle(screen, 'Game Over', 40, WHITE, score)
            font = pygame.font.SysFont('Arial', 20)
            restart_text = font.render('Press R to Restart or Q to Quit', True, WHITE)
            screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 60))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        locked_positions = {}
                        grid = create_grid(locked_positions)
                        current_piece = Piece(shapes[np.random.randint(0, len(shapes))])
                        next_piece = Piece(shapes[np.random.randint(0, len(shapes))])
                        game_over = False
                        fall_time = 0
                        fall_speed = 0.5
                        score = 0
                        level = 1
                        lines_cleared = 0
                    if event.key == pygame.K_q:
                        running = False
            continue

        if paused:
            draw_text_middle(screen, 'Paused', 40, WHITE, score)
            pygame.display.update()
            continue

        gaze_x, frame = eye_tracker.get_gaze_position()
        if gaze_x is not None:
            if prev_gaze_x is None or abs(gaze_x - prev_gaze_x) > 0.1:
                target_x = int(gaze_x * GRID_WIDTH)
                target_x = max(0, min(GRID_WIDTH - len(current_piece.shape[0]), target_x))
                current_piece.x = target_x
                prev_gaze_x = gaze_x

        if eye_tracker.is_blink():
            print("Blink detected! Rotating piece.")  # Debug print
            current_piece.rotate()
            if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                for _ in range(3):
                    current_piece.rotate()

        if fall_time / 1000 >= fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                current_piece.y -= 1
                for y, row in enumerate(current_piece.shape):
                    for x, cell in enumerate(row):
                        if cell:
                            locked_positions[(current_piece.x + x, current_piece.y + y)] = current_piece.color
                lines_cleared += clear_rows(grid, locked_positions)
                if lines_cleared > 0:
                    score += lines_cleared * 10
                    level = score // 50 + 1
                    fall_speed = max(0.1, 0.5 - (level - 1) * 0.05)
                if check_game_over(locked_positions):
                    game_over = True
                    continue
                current_piece = next_piece
                next_piece = Piece(shapes[np.random.randint(0, len(shapes))])

        shape_pos = []
        for y, row in enumerate(current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    shape_pos.append((current_piece.x + x, current_piece.y + y))

        for x, y in shape_pos:
            if y > -1:
                grid[y][x] = current_piece.color

        draw_grid(screen, grid)
        font = pygame.font.SysFont('Arial', 20)
        label = font.render(f"Score: {score}", True, WHITE)
        screen.blit(label, (10, 10))
        level_label = font.render(f"Level: {level}", True, WHITE)
        screen.blit(level_label, (10, 40))
        draw_next_piece(screen, next_piece)

        pygame.display.update()

    eye_tracker.release()
    pygame.quit()

if __name__ == "__main__":
    main()