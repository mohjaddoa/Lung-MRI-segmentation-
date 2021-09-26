import cv2
import numpy as np
from skimage import measure
import imutils

#### read image
readr_img = cv2.imread("test.jpg") 
#### convert to gray
gray = cv2.cvtColor(readr_img, cv2.COLOR_BGR2GRAY)
#### zero matrex
mask = np.zeros((gray.shape),dtype="uint8")
#### bluring image
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#### thresholing
(T, threshInv2) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#### eraion and dilation to remove noise and fill holes
threshInv2 = cv2.erode(threshInv2, None, iterations=2)
threshInv2 = cv2.dilate(threshInv2, None, iterations=4)
threshInv2 = cv2.dilate(threshInv2, None, iterations=2)
#### identifying blobs
labels = measure.label(threshInv2,8,0)
# loop over blobs
for label in np.unique(labels):
    labelMask = np.zeros(threshInv2.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    ### extract only blobs with a speific number of pixels
    if(numPixels <= 18500):
        mask = cv2.add(mask, labelMask)
#### increase size of blob 
mask2 = cv2.dilate(mask, None, iterations=23)
mask = cv2.dilate(mask, None, iterations=7)

#### AND operation
output = cv2.bitwise_and(gray,mask2)
output_second = cv2.bitwise_and(gray,mask)
######## localization 
contores = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contores = imutils.grab_contours(contores)
for coor in contores:
    # obtain the coordinate of blobs
    (x, y, w, h) = cv2.boundingRect(coor)
    ### draw rectangle 
    cv2.rectangle(readr_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #### put title on rectangle
    cv2.putText(readr_img, "Lung", (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)	
############# results #############
#cv2.imshow("mask",mask)
cv2.imshow("full lung",output)
cv2.imshow("output_second",output_second)
cv2.imshow("gray",gray)
cv2.imshow("Image", readr_img)
cv2.waitKey(0)



