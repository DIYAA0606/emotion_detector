# Real-Time Facial Emotion Detector 🎭

A real-time facial emotion detection system that combines a custom-trained CNN with OpenCV face detection to classify emotions from a live webcam feed.

## Features

- **Custom CNN trained from scratch** on the FER2013 dataset (35,000+ labeled facial images) to classify 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Real-time face detection** using OpenCV's Haar Cascade classifier
- **Live webcam inference** — detects faces and overlays predicted emotion labels in real time
- **End-to-end pipeline** — from raw pixel data (FER2013 CSV) to a deployable `.h5` model used for live prediction

## How It Works

### Training (`train_model.py`)
1. Loads the FER2013 dataset from CSV — each row contains space-separated pixel values and an emotion label
2. Reshapes each row into a 48x48 grayscale image, adds a channel dimension `(48, 48, 1)`
3. Normalizes pixel values to `[0, 1]` and one-hot encodes the 7 emotion labels
4. Splits data into training/test sets (80/20)
5. Trains a CNN: two `Conv2D + MaxPooling2D` blocks → `Flatten` → `Dense(128)` with `Dropout(0.5)` → `Dense(7, softmax)`
6. Compiles with Adam optimizer and categorical crossentropy loss, trains for 10 epochs
7. Saves the trained model as `emotion_model.h5`

### Inference (`webcam_emotion.py`)
1. Loads the trained `emotion_model.h5` and OpenCV's pretrained Haar Cascade face detector
2. Captures live video from the webcam
3. Converts each frame to grayscale and detects faces using `detectMultiScale()`
4. For each detected face: crops the region, resizes to 48x48, normalizes, and reshapes to match the model's expected input shape `(1, 48, 48, 1)`
5. Runs the CNN to predict emotion probabilities, takes the highest-confidence class
6. Draws a bounding box and emotion label on the live video feed
7. Press **'q'** to exit

## Tech Stack

**Language:** Python
**ML/DL:** TensorFlow, Keras (CNN architecture)
**Computer Vision:** OpenCV (Haar Cascade face detection, webcam capture)
**Data Processing:** NumPy, Pandas, scikit-learn (train/test split)

## Dataset

[FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — 48x48 grayscale facial images labeled across 7 emotion categories.

## Getting Started

1. Clone the repo
```bash
   git clone https://github.com/DIYAA0606/emotion_detector.git
   cd emotion_detector
```
2. Install dependencies
```bash
   pip install tensorflow opencv-python numpy pandas scikit-learn
```
3. (Optional) Retrain the model — requires `fer2013.csv` in the project directory
```bash
   python train_model.py
```
4. Run real-time detection (uses the pretrained `emotion_model.h5`)
```bash
   python webcam_emotion.py
```

## Why I Built This

Wanted hands-on experience with the full ML pipeline — not just using a pretrained model, but processing raw data, designing and training a CNN from scratch, and deploying it into a real-time computer vision application.
