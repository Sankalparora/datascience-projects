#start the video camera
#put a frame using haarcascade arounf your face
#we want the largest face, so we will sort acc to the area of the frame and it will get stored in an array on our computer.
#crop your face and store in the format of a numpy array
# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etc.


# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np
import os

#KNN CODE

def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]



cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("C:\\Users\\admin\\Desktop\\haarcascade_frontalface_alt.xml")

skip=0
face_data=[]
label=[]
class_id=0 #labels for the given file
names={} #for mapping btw id-name

dataset_path="C:\\Users\\admin\\Desktop\\Datasets\\Images\\data\\"



#data preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):

		#this will give us the mapping between class_id and name
		names[class_id]=fx[:-4] #eg san.npy we are removing .npy
		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)

		#creating labels
		target=class_id*np.ones((data_item.shape[0],))
		class_id+=1
		label.append(target)


#concatenate(beacause here face_data is a list of list , that is why we will have to join them)
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(label,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)
#now we will keep these two x and y into one matrix
trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


#TESTING
while True:
	ret,frame=cap.read()
	grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret==False:
		continue

	
	faces=face_cascade.detectMultiScale(frame,1.5,5)
	

	for face in faces:
		x,y,w,h=face
		#face is a tupple here.
		#extract or crop out the frame
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

		#predict the label
		out=knn(trainset,face_section.flatten())

		#now display the name and the rectangle around it
		pred_name=names[int(out)]


		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) #2 is the thickness
		

	cv2.imshow("Faces",frame)
	key_pressed=cv2.waitKey(1) &0xFF
	if key_pressed==ord('q'):
		break	


cap.release()
cv2.destroyAllWindows()		

