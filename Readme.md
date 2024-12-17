# Ismt Expo 2024

This project contains multiple games that use eye-tracking and other computer vision techniques to control gameplay. The games included are Tetris, CarGame, Flappy, MagicAI, and Sign Language recognition.

## Project Structure

```plaintext
Ismt-Expo/
│
├── CarGame/
│   ├── app1.py
│   ├── haarcascade_eye.xml
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_mcs_nose.xml
│   └── main.py
│
├── Flappy/
│   ├── app1.py
│   └── main.py
│
├── MagicAI/
│   ├── app1.py
│   └── main.py
│
├── main_sign_language/
│   ├── main.py
│   └── sign_language_model.h5
│
├── Sign Language 2/
│   ├── app.py
│   └── sign_language_mnist_model.h5
│
└── Tetris/
    ├── app1.py
    └── main.py
```

## How the Project Works

### Tetris

The Tetris game uses eye-tracking to control the movement of Tetris pieces. The game captures video from the webcam, detects the player's gaze, and moves the Tetris pieces accordingly. Blinking is used to rotate the pieces.

### CarGame

The CarGame uses face and eye detection to control the movement of a car. The player's head position is mapped to the car's position on the screen, allowing the player to steer the car by moving their head.

### Flappy

The Flappy game uses hand detection to control the vertical position of a character. The player raises or lowers their hand to make the character jump or duck.

### MagicAI

The MagicAI project uses a combination of computer vision and TensorFlow to create an invisibility cloak effect. The player can use a red cloak to become invisible on the screen.

### Sign Language Recognition

The Sign Language Recognition projects use hand detection and a pre-trained model to recognize hand signs and display the corresponding letters on the screen.

![Sign Language letters](<amer_sign2.png>)

## How to Run

### Prerequisites

- Python 3.7 or higher
- OpenCV
- Pygame
- Mediapipe
- TensorFlow

### Installation

1. Clone the repository:

```sh
git clone https://github.com/prabinpanta0/Ismt-Expo.git
cd Ismt-Expo
```

2. Install the required packages:

```sh
pip install opencv-python-headless pygame mediapipe tensorflow
```

### Running the Games

#### Tetris

To run the Tetris game, navigate to the `Tetris` directory and run `main.py`:

```sh
cd Tetris
python main.py
```

#### CarGame

To run the CarGame, navigate to the `CarGame` directory and run `main.py`:

```sh
cd CarGame
python main.py
```

#### Flappy

To run the Flappy game, navigate to the  `Flappy` directory and run `main.py` :

```sh
cd Flappy
python main.py
```

#### MagicAI

To run the MagicAI project, navigate to the `MagicAI` directory and run `main.py` :

```sh
cd MagicAI
python main.py
```

#### Sign Language Recognition

To run the Sign Language Recognition project, navigate to the `main_sign_language` directory and run `main.py` :

```sh
cd main_sign_language
python main.py
```

## Packages Used

- `opencv-python-headless`: For computer vision tasks such as face, eye, and hand detection.
- `pygame`: For creating the game interfaces and handling game logic.
- `mediapipe`: For hand and face landmark detection.
- `tensorflow`: For running pre-trained models for tasks like sign language recognition.

## Flowchart

![alt text](<Mermaid.png>)

## License

This project is licensed under the [MIT License](License). See the LICENSE file for details.

## Acknowledgments

- OpenCV for providing the computer vision library.
- Mediapipe for the hand and face detection solutions.
- TensorFlow for the machine learning framework.
- Pygame for the game development library.

Feel free to contribute to this project by submitting issues or pull requests. Enjoy playing the eye-controlled games!
