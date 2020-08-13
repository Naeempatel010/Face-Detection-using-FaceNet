# This is experimental section where the training is done to apply the face embedding to typical machine learning technique like Support Vector Machines
from sklearn.utils import shuffle
import face_recognition
import os
import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
face_embeddings = []
y_label = []

#Change this variable to point to your directory
folder1 = "messi"
folder2 = "ronaldo"

#calculating embedding for all images in the folder
for image_path in os.listdir(folder1):
	image = face_recognition.load_image_file("messi/" + image_path)
	face_embedding = face_recognition.face_encodings(image)
	if face_embedding:
		face_embeddings.append(face_embedding[0])	
		y_label.append("messi")
#calculating embedding for all images in the folder
for image_path in os.listdir(folder2):
	image = face_recognition.load_image_file("ronaldo/"+ image_path)
	face_embedding = face_recognition.face_encodings(image)
	if face_embedding:
		face_embeddings.append(face_embedding[0])
		y_label.append("ronaldo")



#print(face_embeddings)
#print(y_label)

#randomizing the X and Y label
X,y = shuffle(face_embeddings, y_label, random_state = 0)

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


clf = svm.SVC()

#fitting to the SVM classifier
clf.fit(X_train,y_train)

#output the accuracy of the model on test set
print("the accuracy of the model is :")
print(clf.score(X_test,y_test))


#Saving the model in a pickle file
joblib.dump(clf,"svm.pkl")
