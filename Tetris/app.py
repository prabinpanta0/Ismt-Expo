import cv2
import pygame
import numpy as np

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
        # Eye tracking setup
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.prev_eyes_detected = True
        self.blink_detected = False
        self.blink_start_time = None

    def get_gaze_position(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        gaze_x = None
        eyes_detected = False
        pupil_positions = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_color = roi_color[ey:ey + eh, ex:ex + ew]
                # Pupil detection
                _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    (cx, cy), radius = cv2.minEnclosingCircle(c)
                    center = (int(cx), int(cy))
                    pupil_positions.append((x + ex + center[0], y + ey + center[1]))
                    # Draw circle around pupil
                    cv2.circle(eye_color, center, int(radius), (0, 0, 255), 2)
                # Draw rectangle around eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Blink detection logic
        if self.prev_eyes_detected and not eyes_detected:
            # Eyes just closed
            self.blink_start_time = pygame.time.get_ticks()
        elif not self.prev_eyes_detected and eyes_detected:
            # Eyes just opened
            blink_duration = pygame.time.get_ticks() - self.blink_start_time
            if 50 < blink_duration < 400:
                # Blink detected
                self.blink_detected = True
                print("Blink detected")
            else:
                self.blink_detected = False
            self.blink_start_time = None
        else:
            self.blink_detected = False
        self.prev_eyes_detected = eyes_detected
        # Process pupil positions
        if pupil_positions:
            avg_pupil_x = sum([pos[0] for pos in pupil_positions]) / len(pupil_positions)
            gaze_x = avg_pupil_x / frame.shape[1]
            print(f"Gaze position (normalized): {gaze_x}")
        # Show the frame
        cv2.imshow('Eye Tracking', cv2.resize(frame, (400, 300)))
        cv2.waitKey(1)
        return gaze_x

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
            # Remove positions
            for x in range(GRID_WIDTH):
                del locked_positions[(x, y)]
            # Move every row above down
            for pos in sorted(locked_positions, key=lambda k: k[1])[::-1]:
                x, y_pos = pos
                if y_pos < y:
                    new_key = (x, y_pos + 1)
                    locked_positions[new_key] = locked_positions.pop(pos)
    return lines_cleared

def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont('Arial', size)
    label = font.render(text, True, color)
    surface.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - label.get_height() // 2))

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

    running = True
    game_over = False
    while running:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        clock.tick()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        if game_over:
            draw_text_middle(screen, 'Game Over', 40, WHITE)
            font = pygame.font.SysFont('Arial', 20)
            restart_text = font.render('Press R to Restart or Q to Quit', True, WHITE)
            screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 30))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset the game
                        locked_positions = {}
                        grid = create_grid(locked_positions)
                        current_piece = Piece(shapes[np.random.randint(0, len(shapes))])
                        next_piece = Piece(shapes[np.random.randint(0, len(shapes))])
                        game_over = False
                        fall_time = 0
                        fall_speed = 0.5
                        score = 0
                    if event.key == pygame.K_q:
                        running = False
            continue

        # Eye control input
        gaze_x = eye_tracker.get_gaze_position()
        if gaze_x is not None:
            target_x = int(gaze_x * GRID_WIDTH)
            # Ensure the target x is within the grid
            target_x = max(0, min(GRID_WIDTH - len(current_piece.shape[0]), target_x))
            current_piece.x = target_x
            print(f"Piece x-position: {current_piece.x}")

        # Check for blink to rotate piece
        if eye_tracker.is_blink():
            current_piece.rotate()
            # Ensure the rotated piece is in a valid space
            if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                # Revert rotation if not valid
                for _ in range(3):  # Rotate back to original orientation
                    current_piece.rotate()

        # Piece falls over time
        if fall_time / 1000 >= fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                current_piece.y -= 1
                # Lock the piece
                for y, row in enumerate(current_piece.shape):
                    for x, cell in enumerate(row):
                        if cell:
                            locked_positions[(current_piece.x + x, current_piece.y + y)] = current_piece.color
                # Check for line clears
                lines_cleared = clear_rows(grid, locked_positions)
                if lines_cleared > 0:
                    score += lines_cleared * 10
                    print(f"Score: {score}")
                    # Increase speed every 50 points
                    fall_speed = max(0.1, fall_speed - (score // 50) * 0.05)
                # Check for game over
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

        # Draw current piece on grid
        for x, y in shape_pos:
            if y > -1:
                grid[y][x] = current_piece.color

        # Draw grid and score
        draw_grid(screen, grid)
        font = pygame.font.SysFont('Arial', 20)
        label = font.render(f"Score: {score}", True, WHITE)
        screen.blit(label, (10, 10))

        pygame.display.update()

    eye_tracker.release()
    pygame.quit()

if __name__ == "__main__":
    main()