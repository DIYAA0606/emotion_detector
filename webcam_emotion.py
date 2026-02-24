from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Labels in the same order as the model's output
emotion_labels = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Sad', 'Surprise', 'Neutral'
]


def load_emotion_model(path: str = "emotion_model.h5"):
    return load_model(path)


def open_camera(index: int = 0):
    return cv2.VideoCapture(index)


def preprocess_face(gray, x, y, w, h):
    # crop
    roi = gray[y:y + h, x:x + w]
    # resize to 48x48
    roi = cv2.resize(roi, (48, 48))
    # normalize to [0,1]
    roi = roi.astype("float32") / 255.0
    # reshape to (1,48,48,1)
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi


def annotate_frame(frame, x, y, w, h, label: str):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


def main():
    # open webcam
    cap = open_camera(0)

    model = load_emotion_model()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            break

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # crop, resize, normalize, reshape
            roi = preprocess_face(gray, x, y, w, h)

            # predict
            preds = model.predict(roi)
            max_index = np.argmax(preds[0])
            emotion_label = emotion_labels[max_index]

            # draw rectangle + label
            annotate_frame(frame, x, y, w, h, emotion_label)

        # show frame
        cv2.imshow('Emotion Detection', frame)

        # break if q pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release camera
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
