import cv2
import os 
import argparse

parser = argparse.ArgumentParser(description="well nothing much")

parser.add_argument('folder', metavar = 'folder', type=str,help="Specify the folder name for the storing the images")

parser.add_argument('count',metavar ='count', type=int,help="Number of images")

args = parser.parse_args()

count = args.count
folder = args.folder
try:
	abs_path = os.path.join(os.getcwd(),folder)
	os.mkdir(abs_path)
except OSError as error:
	print(error)
	exit(0)


cap = cv2.VideoCapture(0)

cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
if not cap.isOpened():
	print("error opening camera")
	exit(0)

while True:
	if count <= 0:
		break
	ret, frame = cap.read()
	if frame is None:
		print("No captured frame")
		break
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.equalizeHist(frame_gray)
	
	faces = cascade.detectMultiScale(frame_gray)
	for (x,y,w,h) in faces:
		count = count - 1
		print(count)
		cv2.imwrite(os.path.join(os.getcwd(),folder)+ "/"+ str(count)+ ".jpg",frame[y:y+h,x:x+w])
		frame_gray = cv2.rectangle(frame_gray, (x,y),(x+h,y+h),2)
	
	cv2.imshow("myface",frame_gray)
	
	if cv2.waitKey(1) == 27:
		break


cv2.destroyAllWindows()
cap.release()

