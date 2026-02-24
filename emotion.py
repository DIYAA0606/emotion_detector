import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
data = pd.read_csv("fer2013.csv")

X = []
y = []

for index, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype=np.uint8)
    image = pixels.reshape(48, 48)
    
    X.append(image)
    y.append(row['emotion'])

X = np.array(X)
y = np.array(y)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)
# Add channel dimension
X = X.reshape(-1, 48, 48, 1)

# Normalize
X = X / 255.0

# Convert labels to categorical (7 emotions)
y = to_categorical(y, 7)

print("After reshape:", X.shape)
print("After categorical:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
model.save("emotion_model.h5")
print("Model saved successfully!")
