# Mica-The-Smart-Doll-Capstone-Project-
In this capstone project I have developed computer vision specific features like emotion recognition, face detection, and object detection using Python, OpenCV, and deep learning models.


---

## 1. Emotion Detection Using Facial Expressions

The system first detects faces using OpenCV’s Haar Cascade classifier, preprocesses the detected face region, and then predicts the emotion using a pre-trained CNN model.

---

## Key Features

* Real-time emotion detection using webcam input
* Face detection using Haar Cascade (`haarcascade_frontalface_default.xml`)
* CNN-based emotion classification with 7 emotion classes:

  * Angry
  * Disgusted
  * Fearful
  * Happy
  * Neutral
  * Sad
  * Surprised
    
* Grayscale image preprocessing and resizing to 48×48
* Bounding box and emotion label displayed on detected faces

---

## Model Architecture

* Convolutional Neural Network (CNN) built using Keras
* Pre-trained weights loaded from `model.h5`

---

## Workflow

1. Capture image from webcam
2. Convert frame to grayscale
3. Detect face regions
4. Resize and normalize face image
5. Predict emotion using trained CNN
6. Display emotion label on the image

---

## Tools & Technologies

* Python
* OpenCV
* TensorFlow / Keras
* NumPy


Here is a **matching, clean README section** for **Face Recognition**, written in the **same style and level of detail** as your emotion detection section.

---

## 2. Face Recognition Using Computer Vision

The system captures images from a webcam, detects human faces, and identifies individuals by comparing facial features against known reference images stored locally.

---

## Key Features

* Real-time face recognition using webcam input
* Face detection using the `face_recognition` library
* Support for multiple known individuals via reference images
* Bounding box and name label displayed on detected faces

---

## Recognition Method

* Encodings from live images are compared with stored reference encodings
* Euclidean distance is used to determine the closest match
* Faces are labeled as **Unknown** if no match meets the similarity threshold

---

## Workflow

1. Load reference images from the local repository
2. Encode known faces into feature vectors
3. Capture image from webcam
4. Detect faces in the captured image
5. Encode detected faces
6. Compare encodings with known faces
7. Display identity label on the image

---

## Tools & Technologies

* Python
* OpenCV
* face_recognition
* NumPy

