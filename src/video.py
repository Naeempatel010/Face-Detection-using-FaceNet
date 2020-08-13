#this file for processing the face_recognition on Videos. This py file is used to detect "Gabbar" from Sholay in his iconic "Kitne admi the?"

import face_recognition
import cv2
import numpy as np
fourcc  = cv2.VideoWriter_fourcc(*'XVID')


#out = cv2.VideoWriter('output.avi',fourcc, , (640,480))

image = "sholay/gabbar.jpg"
gabbar_image = face_recognition.load_image_file(image)
gabbar_face_encoding = face_recognition.face_encodings(gabbar_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
       gabbar_face_encoding
    ]
known_face_names = ["Gabbar"]

video_location  = 'sholay/kitne.mp4'
cap = cv2.VideoCapture(video_location)

out = cv2.VideoWriter("output.avi",fourcc,cap.get(5),(int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
	ret, frame = cap.read() 
	if frame is None:
		break
	rgb_frame = frame[:, :, ::-1]
	face_locations = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    # Loop through each face in this frame of video
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

		name = "Unknown"
		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]

		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	out.write(frame)
	#print("working")
	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF== ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()



