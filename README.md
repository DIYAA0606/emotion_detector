# 🎯 Real-Time Face Detection using OpenCV

This project implements real-time face detection using OpenCV's pre-trained Haar Cascade classifier and a live webcam feed.

It captures video from the system webcam, detects faces in each frame, and draws bounding boxes around detected faces.

---

## 🚀 Features

- Real-time face detection
- Uses Haar Cascade classifier
- Webcam integration
- Draws bounding boxes around detected faces
- Lightweight and beginner-friendly implementation

---

## 🛠️ Technologies Used

- Python
- OpenCV (cv2)

---

## 📂 How It Works

1. Loads a pre-trained Haar Cascade model for frontal face detection.
2. Captures live video using the webcam.
3. Converts each frame to grayscale for faster processing.
4. Detects faces using `detectMultiScale()`.
5. Draws red rectangles around detected faces.
6. Press **'q'** to exit the application.

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
