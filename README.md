# Air-Canvas

This project is an air canvas that allows users to draw on a virtual whiteboard using their finger. It uses OpenCV for computer vision, MediaPipe for hand tracking, and various libraries for managing points and drawing on the canvas.

## Features

- Draw with different colors: Blue, Green, Red, and Yellow.
- Clear the canvas.
- Undo and redo the last drawing actions.
- Real-time hand tracking for drawing using finger movements.

## Requirements

- Python 3.11
- OpenCV
- NumPy
- MediaPipe

## Installation

1. **Clone the repository:**
    ```bash
    https://github.com/GuptaRaj007/Air-Canvas.git
    cd air-canvas
    ```

2. **Install the required libraries:**
    ```bash
    pip install opencv-python-headless numpy mediapipe
    ```

## Usage

1. **Run the canvas script:**
    ```bash
    python canvas.py
    ```

2. **Use the following buttons on the canvas:**
    - **CLEAR**: Clears the entire canvas.
    - **BLUE, GREEN, RED, YELLOW**: Select the color to draw with.
    - **UNDO**: Undo the last drawing action.
    - **REDO**: Redo the previously undone drawing action.

## Code Overview

The script initializes the MediaPipe hand tracking module and sets up the OpenCV window with the drawing canvas and buttons for color selection, clear, undo, and redo functionalities. 

### Main Components:

- **MediaPipe Hands**: Used for detecting and tracking hand landmarks.
- **OpenCV**: Used for capturing video from the webcam, drawing rectangles and text, and creating the drawing canvas.
- **NumPy**: Used for creating the blank canvas.
- **Deque**: Used for managing drawing points and history for undo and redo functionalities.

### Drawing Logic

- Hand landmarks are detected using MediaPipe, and the index finger tip coordinates are used for drawing.
- The canvas includes buttons for selecting drawing colors, clearing the canvas, and undoing/redoing the last actions.
- The points drawn on the canvas are stored in deques, and history states are managed using a stack for undo and redo operations.

## Example

After running the script, the webcam feed will display a window with the drawing canvas. Use your index finger to draw on the canvas and interact with the buttons for various functionalities.


