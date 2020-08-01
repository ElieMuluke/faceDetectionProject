# importing library
import cv2
from random import randrange

# Detect from image
# Load some pre-trained data on face frontal from opencv repository (https://github.com/opencv/opencv/tree/master/data/haarcascades)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect face in
img = cv2.imread('us.jpg')

# convert it to grayScaled (must)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)

# draw rectangle
for (x, y, w, h) in face_coordinates:
  cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)
# Display
cv2.imshow('Face Detector', img)
cv2.waitKey()

print("Code Image completed")
