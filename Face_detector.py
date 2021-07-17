# importing library
import cv2
from random import randrange

choice = input("Recognize from image or webcam? ")
choice.lower()


def image():
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


def webcam():
    # Detect from image
    # Load some pre-trained data on face frontal from opencv repository (https://github.com/opencv/opencv/tree/master/data/haarcascades)
    trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Capture video from webcam P.S.: # we can also use a video file as input
    webcam = cv2.VideoCapture(0)

    # Iterate forever over frames
    while True:
        # Read current frames from the webcam
        successful_frame_read, frame = webcam.read()

        # Convert to grayscale
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_coordiantes = trained_face_data.detectMultiScale(grayscaled_img)

        # Drawing rectangles
        for (x, y, w, h) in face_coordiantes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

        # Display the image with the squares on the frame
        cv2.imshow('Face detection webcam', frame)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            break
    # after quitting release camera
    webcam.release()

    print("Video code completed")


if choice == "image":
    image()
if choice == "webcam":
    webcam()
