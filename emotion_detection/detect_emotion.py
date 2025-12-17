import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import pandas as pd
from keras import models, layers, optimizers
import warnings


warnings.filterwarnings('ignore')

model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))

#load trained model
model.built = True
model.load_weights("model.h5")


# mapping class labels 
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not vid.isOpened():
    raise RuntimeError("Could not open webcam. Check permissions or if another app is using it.")


facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



while True:
    ok, frame = vid.read()
    if not ok:
        print("Failed to read frame from camera.")
        break

    frame = cv2.flip(frame, 1) 

    cv2.imshow("image", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        cv2.imwrite("emotion.jpg", frame)
        break

vid.release()
cv2.destroyAllWindows() 
    
frame = cv2.imread("emotion.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)
    maxindex = int(np.argmax(prediction))
    print(emotion_dict[maxindex])
    cv2.putText(frame, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('Emotion', cv2.resize(frame,(640, 480),interpolation = cv2.INTER_CUBIC))

cv2.waitKey(0)
cv2.destroyAllWindows()