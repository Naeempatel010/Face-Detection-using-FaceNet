#This file is used for detecting people using the webcam. This file initially loks for a embedding variable named model to retrieve a variable consisting of embedding. If it does not exist then it creates one. Done primarily for optimization.

import face_recognition
import cv2
import numpy as np
from os import path
import pickle5 as pickle
video_capture = cv2.VideoCapture(0)


#generate encoding from the images
def gen_encodings():
	image = "criminals/naeem.png"
	naeem_image = face_recognition.load_image_file(image)
	naeem_face_encoding = face_recognition.face_encodings(naeem_image)[0]

	# Load a second sample picture and learn how to recognize it.
	image = "criminals/yashveer.jpg"	
	yash_image = face_recognition.load_image_file(image)
	yash_face_encoding = face_recognition.face_encodings(yash_image)[0]
	
	image = "criminals/sai.png"
	sai_image = face_recognition.load_image_file(image)
	sai_face_encoding = face_recognition.face_encodings(sai_image)[0]

# Create arrays of known face encodings and their names
	known_face_encodings = [naeem_face_encoding,yash_face_encoding,sai_face_encoding]
	with open("model.pkl", "wb") as f:
		pickle.dump(known_face_encodings,f)
	return known_face_encodings

known_face_encodings = []

if not path.exists('model.pkl'):
	known_face_encodings = gen_encodings()
else:
	with open("model.pkl","rb") as f:
		known_face_encodings = pickle.load(f)

known_face_names = ["Naeem Patel","Yashveer Singh","Sai Reddy"]

while True:
    
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"


        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

