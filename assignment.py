import cv2 as cv
import numpy as np
import sys
from pathlib import Path
import os
import re

def knn(digits, folder, fileName, imageNum, fileExt):
    modelsFolder = "models/"
    #modelsFolder = "/home/student/kay_men_yap_19257442/models/"
    # KNN model loading and testing code obtained from https://docs.opencv.org/3.4.2/d8/d4b/tutorial_py_knn_opencv.html
    # Last accessed on 24/10/2020
    with np.load(modelsFolder+'knn_data.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
        knn = cv.ml.KNearest_create()
        knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

        # Convert the list of digit images into a numpy array and reshape it to be a 2d array containing 1d float32 array of the digits
        digit_array = np.array(digits).reshape(-1, 1120).astype(np.float32)
        ret,result,neighbours,dist = knn.findNearest(digit_array,k=3)
        
        #Convert the result into a list of strings
        result_list = [str(i[0]) for i in list(result.astype(np.int32))]
        
        # Write house number obtained to file
        with open(folder+'House'+imageNum+'.txt', "w") as houseFile:
            houseFile.write("Building "  + "".join(result_list))
        #print("Building " + "".join(result_list))

def preprocess(image, folder, fileName, imageNum, fileExt):
    original = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Bilateral filter code obtained from https://docs.opencv.org/3.4.2/d4/d13/tutorial_py_filtering.html
    # Last accessed on 24/10/2020
    gray = cv.bilateralFilter(gray,9,75,75)
    
    # Thresholding with Otsu's Binarsization code obtained from https://docs.opencv.org/3.4.2/d7/d4d/tutorial_py_thresholding.html
    # Last accessed on 25/10/2020
    (thresh, binary) = cv.threshold(gray, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #cv.imwrite(folder + fileName +"-binary" +fileExt, binary)

    # Contour code obtained from https://docs.opencv.org/3.4.2/d4/d73/tutorial_py_contours_begin.html 
    # Last accessed on 25/10/2020
    im2, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    '''
    # For the purpose of displaying all contours detected initially
    for cnt in contours:
        # Bounding rectangle and contour area code obtained from https://docs.opencv.org/3.4.2/dd/d49/tutorial_py_contour_features.html
        # Last accessed on 25/10/2020
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imwrite(folder + fileName + "-contours-unfiltered" +fileExt, image)
    '''
    

    # Getting the image dimensions, area and max_x and max_y for filtering
    height, width = image.shape[:2]
    image_size = height * width
    max_x = width - 1
    max_y = height - 1
    
    image = original.copy()
    validContours = list()
    maxArea = 0
    maxIndex = 0
    index = 0
    # Filter out any contour that is too big or too small or doesn't have a height to width ratio of the bounding rectangle that is within the range specified 
    # below and any contour that is too near the edge of the image
    for cnt in contours:
        # Bounding rectangle and contour area code obtained from https://docs.opencv.org/3.4.2/dd/d49/tutorial_py_contour_features.html
        # Last accessed on 25/10/2020
        x,y,w,h = cv.boundingRect(cnt)
        rectArea = h * w
        contourArea = cv.contourArea(cnt)
        areaRatio = rectArea / image_size
        # Filter the contours that are too big or too small, and the bounding rectangles are not a vertical rectangle t
        if ((areaRatio > 0.0001) and (rectArea < image_size / 4) and h / w > 1.3 and h / w < 3.5 and x > 40 and y > 20 and x + w < max_x - 40 and y + h < max_y - 20):
            validContours.append(cnt)
            #cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            # Updating the biggest valid contour
            if(rectArea > maxArea):
                maxArea = rectArea
                maxIndex = index
            index += 1
    #cv.imwrite(folder + fileName + "-contours-filter1" +fileExt, image)
    
    # Find all contours that are at least 35% of the size of the largest contour and 
    # its centre y value is near to the largest contour's centre y value with a difference
    # of less than 25% of the height of largest contour
    image =  original.copy()
    filteredList = list()
    biggestCnt = validContours[maxIndex]
    largest_x, largest_y, largest_w, largest_h = cv.boundingRect(biggestCnt)
    center_y_largest = (largest_y + largest_h) / 2
    acceptedOffset = (largest_y + largest_h - center_y_largest) / 2
    for cnt in validContours:
        x,y,w,h = cv.boundingRect(cnt)
        center_y = (y + h) / 2
        rectArea = h * w
        contourArea = cv.contourArea(cnt)
        y_centre_offset = abs(center_y_largest - center_y)
        areaRatio = rectArea / maxArea
        #print("AreaRatio: " + str(areaRatio) + ", Y_Center_Offset: " + str(y_centre_offset))
        if(areaRatio> 0.35 and y_centre_offset < acceptedOffset):
            filteredList.append(cnt)
            #cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #cv.imwrite(folder + fileName + "-contours-filter2" +fileExt, image)

    # Find all contours that are not within another contour's bounding rectangle
    digitContours = list()
    approvedIndexList = list()
    for i,cnt1 in enumerate(filteredList):
        x1,y1,w1,h1 = cv.boundingRect(cnt1)
        valid = True
        for j,cnt2 in enumerate(filteredList):
            if( i != j):
                x2,y2,w2,h2 = cv.boundingRect(cnt2)
                # Check if the cnt2's bounding rectangle is within cnt1's bounding rectangle
                if(x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1):
                    valid = False
        if valid:
            approvedIndexList.append(i)
    
    image =  original.copy()
    for i in approvedIndexList:
        cnt = filteredList[i]        
        x,y,w,h = cv.boundingRect(cnt)
        digitContours.append(filteredList[i])
        #cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #cv.imwrite(folder + fileName + "-contours-filter3" +fileExt, image)
    
    # Find the bounding box coordinates to crop the detected area by finding for the min and max values of x and y 
    # from the bounding rectangles of all contours in digit contours list
    image =  original.copy()
    firstCnt = digitContours[0]
    first_x, first_y, first_w, first_h = cv.boundingRect(firstCnt)
    min_x = first_x
    max_x = first_x + first_w
    min_y = first_y
    max_y = first_y + first_h
    extractedDigits = list()
    for cnt in digitContours:        
        x,y,w,h = cv.boundingRect(cnt)
        if(x < min_x):
            min_x = x
        if(x+w > max_x):
            max_x = x+w
        if(y < min_y):
            min_y = y
        if(y+h > max_y):
            max_y = y+h
    
    # Calculuate the width and height of the bounding box of the detected area
    crop_w = max_x - min_x
    crop_h = max_y - min_y
    
    # Crop the detected area from the original image
    cropped = original[min_y:max_y, min_x:max_x]

    # Write the bounding box details of the detected area to txt file as required in assignment spec
    with open(folder+'BoundingBox'+ imageNum+'.txt', "w") as boxFile:
        boxFile.write(str(min_x) + ',' + str(min_y) + ',' + str(crop_w) + ',' + str(crop_h))
    
    
    #cv.imwrite(folder + fileName + "-contours-final" +fileExt, image)

    # Preprocess the cropped image to perform digit extraction
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray,9,75,75)
    (thresh, binary) = cv.threshold(gray, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Find the contours in the cropped(detected area) binary image
    im2, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Get the detected area image's area
    croppedArea = cropped.shape[0] * cropped.shape[1]
    
    # Filter out the small and inner contours 
    contours = remove_small_contours(contours, croppedArea)
    contours = remove_inner_contours(contours)

    # Contour sorting using x coordinate value so that the digit contours are in order from left to right of the image
    #    
    # Last accessed on 24/10/2020
    boundingBoxes = [cv.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda b:b[1][0], reverse=False))

    # Loop through the contours to get the individual digits and append to extractedDigits
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        
        #print("Contour Area: " + str(contourArea) + ", Rect Area: " + str(rectArea) + ", X: " + str(x) + ", Y: " + str(y) + ", W: " + str(w) + ", H: " + str(h) + ", Approx: " + str(len(approx)))
        #cv.drawContours(cropped, [cnt], 0, (0,255,0), 3)

        # Extract the digit from binary image
        digit = binary[y:y+h, x:x+w]
        
        # Adapted code to add black borders to the cropped digit images from https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
        # Last accessed on 24/10/2020
        padded = cv.copyMakeBorder(digit, 4,4,4,4, cv.BORDER_CONSTANT, None, [0,0,0])

        # Image resizing code obtained and adapted from https://docs.opencv.org/3.4.2/da/d6e/tutorial_py_geometric_transformations.html
        # Last accessed on 25/10/2020
        # Resize the digit to be the same size as the training digits used for training the KNN model
        digit = cv.resize(padded, (28,40), interpolation = cv.INTER_CUBIC)
        extractedDigits.append(digit)
        cv.rectangle(cropped,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imwrite(folder + 'DetectedArea' + imageNum + '.jpg', cropped)
    #cv.imwrite(folder + fileName + "-cropped-contours" +fileExt, cropped)

    return extractedDigits

def remove_small_contours(contours, imageArea):
    approvedContours = list()
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        rectArea = h * w
        areaRatio = rectArea / imageArea
        if(areaRatio > 0.02):
            approvedContours.append(cnt)

    return approvedContours

def remove_inner_contours(contours):
    approvedContours = list()
    approvedIndexList = list()
    for i,cnt1 in enumerate(contours):
        valid = True
        x1,y1,w1,h1 = cv.boundingRect(cnt1)
        for j,cnt2 in enumerate(contours):
            if( i != j):
                x2,y2,w2,h2 = cv.boundingRect(cnt2)
                # Check if the cnt2's bounding rectangle is within cnt1's bounding rectangle
                if (x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1):
                    valid = False
        if valid:
            approvedIndexList.append(i)
    
    for i in approvedIndexList:
        approvedContours.append(contours[i])
    return approvedContours

def main():
    if(len(sys.argv) == 1):
        #outputFolder = "/home/student/kay_men_yap_19257442/output/"
        outputFolder = "output/"
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        #folder = "/home/student/test/"
        folder = "test/"
        for filename in os.listdir(folder):
            # Sanity check if filename is a file in case I used the /home/student/train directory for 
            # training images to test with
            if not os.path.isdir(folder+filename):
                path = Path(filename)
                #print(filename)
                image = cv.imread(folder+filename)
                imageNum = re.findall('[0-9]+$', path.stem)[0]
                extractedDigits = preprocess(image.copy(), outputFolder, path.stem, imageNum, path.suffix)
                knn(extractedDigits, outputFolder, path.stem, imageNum, path.suffix)
    else:
        print("Please run the program with 'python assignment.py'")

if(__name__ == '__main__'):
    main()