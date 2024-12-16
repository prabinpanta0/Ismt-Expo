import cv2
import numpy as np
import tensorflow as tf
import time

# -----------------------------
# PART 1: Camera & Initial Setup
# -----------------------------
cap = cv2.VideoCapture(0)
time.sleep(2)  # Allow the camera to warm up

# Define HSV color range for the cloak (here using red as an example)
# You should adjust these values based on your cloak's color.
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Threshold for deciding if cloak is detected (in terms of mask area)
CLOAK_AREA_THRESHOLD = 5000

# Morphological kernel sizes for noise removal and smoothing
open_kernel = np.ones((3,3), np.uint8)
close_kernel = np.ones((5,5), np.uint8)

# Initialize background and detection flag
background = None
cloak_detected = False

# -----------------------------
# PART 2: TensorFlow No-Op Model
# -----------------------------
# A trivial model that just casts the input to float32, fulfilling TF integration requirement.
input_layer = tf.keras.Input(shape=(None, None, 3), dtype=tf.uint8)
x = tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(input_layer)
model = tf.keras.Model(inputs=input_layer, outputs=x)

@tf.function
def process_frame_with_tf(frame):
    output = model(frame[None, ...])
    return tf.squeeze(output, axis=0)

# -----------------------------
# PART 3: Processing Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally if desired for more "mirror-like" experience
    frame = np.flip(frame, axis=1)

    # Optional: Denoise the frame to reduce random noise and get a cleaner mask
    frame_denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # Process frame with TensorFlow model (no-op here)
    tf_frame = tf.convert_to_tensor(frame_denoised, dtype=tf.uint8)
    tf_processed_frame = process_frame_with_tf(tf_frame).numpy().astype(np.uint8)

    # Convert to HSV
    hsv = cv2.cvtColor(tf_processed_frame, cv2.COLOR_BGR2HSV)

    # Create the mask for the chosen cloak color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Morphological operations to clean up the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Smooth the edges of the mask
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    # Check if cloak is detected
    cloak_area = cv2.countNonZero(mask)
    if cloak_area > CLOAK_AREA_THRESHOLD:
        cloak_detected = True
    else:
        cloak_detected = False

    # Update background if cloak not detected
    if not cloak_detected:
        if background is None:
            # Initialize background if not set
            background = tf_processed_frame.copy()
        else:
            # Update background using a running average for adaptability
            alpha = 0.1
            background = cv2.addWeighted(tf_processed_frame, alpha, background, 1 - alpha, 0)

    if background is None:
        # If we don't have a background yet, just display the current frame
        cv2.imshow("Invisibility Cloak", tf_processed_frame)
    else:
        # Convert mask to float for alpha blending
        mask_f = mask.astype(np.float32) / 255.0

        # Convert images to float for blending
        cloak_area_img = background.astype(np.float32)
        non_cloak_area = tf_processed_frame.astype(np.float32)

        # Alpha blend:
        # output_pixel = mask_f * cloak_area_img + (1 - mask_f) * non_cloak_area
        output = (cloak_area_img * mask_f[:,:,None] + non_cloak_area * (1 - mask_f[:,:,None])).astype(np.uint8)

        cv2.imshow("Invisibility Cloak", output)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
