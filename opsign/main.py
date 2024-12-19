import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

# Define the classes
# ...existing code...
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
# ...existing code...

# Reconstruct the exact model architecture used during training
model = Sequential()
model.add(Conv2D(128, kernel_size=(5,5), strides=1, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(64, kernel_size=(2,2), strides=1, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Conv2D(32, kernel_size=(2,2), strides=1, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='same'))
# ...existing code...
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))
# ...existing code...

# Load the weights
model.load_weights('sign_language.h5')

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands

# ...rest of your code...

# ...rest of your code...
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# List of classes (letters or words) corresponding to the model's output

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box of the hand
            h, w, c = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            # Crop the hand region
            hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if hand_img.size == 0:
                continue

            # # Resize and normalize the hand image
            # hand_img = cv2.resize(hand_img, (64, 64))
            # hand_img = hand_img / 255.0
            # hand_img = np.expand_dims(hand_img, axis=0)

            # # Predict the hand sign
            # prediction = model.predict(hand_img)
            # predicted_class = np.argmax(prediction)
            # predicted_letter = classes[predicted_class]
            
            # ...existing code...

            # Resize and normalize the hand image
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(hand_img, (28, 28))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)
            hand_img = np.expand_dims(hand_img, axis=-1)

            # Predict the hand sign
            prediction = model.predict(hand_img)
            predicted_class = np.argmax(prediction)
            predicted_letter = classes[predicted_class]

            # ...existing code...
            # Display the predicted letter on the frame
            cv2.putText(frame, f'Letter: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Sign Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
