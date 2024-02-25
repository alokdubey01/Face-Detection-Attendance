import cv2
import numpy as np
import face_recognition

# Load the image
imgelon = face_recognition.load_image_file('alok.jpg')
imgelon = cv2.cvtColor(imgelon, cv2.COLOR_BGR2RGB)

# Find face location and draw rectangle
face_location = face_recognition.face_locations(imgelon)[0]
copy = imgelon.copy()
cv2.rectangle(copy, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (255, 0, 255), 2)

# Display the image with the rectangle
cv2.imshow('Copy', copy)
cv2.imshow('Elon', imgelon)
cv2.waitKey(0)

# Encode the face
train_elon_encoding = face_recognition.face_encodings(imgelon)[0]

# Load another image for testing
test = face_recognition.load_image_file('alok.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]

# Compare the encodings
print(face_recognition.compare_faces([train_elon_encoding], test_encode))