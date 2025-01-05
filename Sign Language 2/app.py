import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('sign_language.h5')

# Define a dictionary to map class indices to letters (A-Y excluding J)
class_map = {i: chr(65 + i) if i < 9 else chr(66 + i) for i in range(24)}

# Start video capture
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """
    Preprocess the captured frame for model input.
    Convert to grayscale, resize to 28x28, normalize, and expand dimensions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (28, 28))           # Resize to 28x28 for the model
    rgb = cv2.merge([resized, resized, resized])   # Convert grayscale to RGB
    normalized = rgb / 255.0                       # Normalize pixel values
    expanded = np.expand_dims(normalized, axis=0)  # Add batch dimension
    return expanded

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI) for hand detection
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Preprocess the ROI and make a prediction
    input_data = preprocess_frame(roi)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    predicted_letter = class_map.get(predicted_class, "?")

    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {predicted_letter}", (120, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video frame
    cv2.imshow("Hand Sign Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
