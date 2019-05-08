#import cv2
#import sys
#
#cascPath = "haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascPath)
#
#video_capture = cv2.VideoCapture(0)
#
#while True:
#    # Capture frame-by-frame
#    ret, frame = video_capture.read()
#
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#    faces = faceCascade.detectMultiScale(
#        gray,
#        scaleFactor=1.3,
#        minNeighbors=5,
#        minSize=(30, 30),
#        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
#        flags=cv2.CASCADE_SCALE_IMAGE
#    )
#
#    # Draw a rectangle around the faces
#    for (x, y, w, h) in faces:
#        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#        count += 1
#
#    # Display the resulting frame
#    cv2.imshow('Video', frame)
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
## When everything is done, release the capture
#video_capture.release()
#cv2.destroyAllWindows()
#
#
#
#
#
#
#
#

# Import OpenCV2 for image processing
import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
face_id=input('enter your id')
# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize sample face image
count = 0

assure_path_exists("dataset/")

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count>=30:
        print("Successfully Captured")
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()

