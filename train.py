import cv2 as cv
import numpy as np
import sys
from pathlib import Path
import os

# KNN training and testing code obtained and adapted from from https://docs.opencv.org/3.4.2/d8/d4b/tutorial_py_knn_opencv.html
# Last accessed on 24/10/2020
def knn(train_data, train_labels, folder):
    test_labels = train_labels.copy()

    knn = cv.ml.KNearest_create()
    knn.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    # Reuse the training data and labels as the testing data and labels to test accuracy of classification
    ret,result,neighbours,dist = knn.findNearest(train_data,k=3)
    # Check the accuracy of classification by comparing the results and the test labels
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print("K = 3, Accuracy: " +str(accuracy) )
    np.savez(folder+'knn_data.npz',train=train_data, train_labels=train_labels)

def preprocess(image):
    gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    (thresh, binary) = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary

def main():
    if(len(sys.argv) == 1):  
        outputFolder = "/home/student/kay_men_yap_19257442/models/"
        #outputFolder = "models/"
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        #folder = "train/"
        folder = "/home/student/train/"
        digitList = list()
        labelList = list()
        count = 0
        for subfolder in os.listdir(folder):
            if os.path.isdir(folder+subfolder):
                for filename in os.listdir(folder+subfolder):
                    path = Path(filename)
                    #print(filename)
                    # Get the train label from filename 
                    labelList.append(filename[5])
                    image = cv.imread(folder+subfolder+"/"+filename)
                    # Preprocess the image and reshape it into a 1d array
                    image = preprocess(image).reshape(-1,1120).squeeze()
                    digitList.append(image)
        
        train_array = np.array(digitList).astype(np.float32)
        train_labels = np.array([[i] for i in labelList]).astype(np.int32)
        knn(train_array, train_labels, outputFolder)
    else:
        print("Please run the program with 'python train.py'")

if(__name__ == '__main__'):
    main()