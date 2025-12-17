import os
import cv2
import numpy as np
import face_recognition




# Load known faces
known_encodings = []
known_names = []

repo_dir = "pic_repository"
for file in os.listdir(repo_dir):
    if file.startswith("."):
        continue
    img = face_recognition.load_image_file(os.path.join(repo_dir, file))
    encs = face_recognition.face_encodings(img, num_jitters=0) 
    if len(encs) == 0:
        print("No face found in:", file)
        continue
    known_encodings.append(encs[0])
    known_names.append(os.path.splitext(file)[0])

print("Loaded faces:", known_names)





# Open camera and capture on SPACE
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not working.")
    exit()

captured = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Camera (Press SPACE to capture, Q to quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        captured = frame.copy()
        cv2.imwrite("captured.jpg", captured)
        print("Captured image saved as captured.jpg")
        break
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if captured is None:
    print("No image captured.")
    exit()





# Recognize faces from the captured image
rgb = captured[:, :, ::-1]
rgb = np.ascontiguousarray(rgb)  

locations = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, locations, num_jitters=0) 

if len(encodings) == 0:
    print("No face detected in captured image.")
    exit()

for enc, (top, right, bottom, left) in zip(encodings, locations):
    name = "Unknown"

    if len(known_encodings) > 0:
        dists = face_recognition.face_distance(known_encodings, enc)
        best = int(np.argmin(dists))
        if dists[best] < 0.6:
            name = known_names[best]

    cv2.rectangle(captured, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(captured, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    print("Recognized:", name)





cv2.imshow("Result", captured)
cv2.waitKey(0)
cv2.destroyAllWindows()

