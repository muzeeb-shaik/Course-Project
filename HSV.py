import cv2
import numpy as np
import csv
import math


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def nothing(x):
    pass

# Creating Trackbars for the HSV slider adjustment
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 0, 255, nothing)

# Getting data from Image
frame = cv2.imread('COVID-19.png')
scale_percent = 60 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
frame = resized
while(True):

	#### To create blurred frame (Choose which kind of blur is required)####
	blurred_frame=cv2.medianBlur(frame, 5)
	#blurred_frame=cv2.GaussianBlur(frame, (5,5),0)

	hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
	#print(hsv.shape)

	l_h = cv2.getTrackbarPos("L - H", "Trackbars")
	l_s = cv2.getTrackbarPos("L - S", "Trackbars")
	l_v = cv2.getTrackbarPos("L - V", "Trackbars")
	u_h = cv2.getTrackbarPos("U - H", "Trackbars")
	u_s = cv2.getTrackbarPos("U - S", "Trackbars")
	u_v = cv2.getTrackbarPos("U - V", "Trackbars")

	lower_value = np.array([l_h,l_s,l_v])
	upper_value = np.array([u_h,u_s,u_v])

	mask = cv2.inRange(hsv, lower_value, upper_value)
	result = cv2.bitwise_and(blurred_frame,blurred_frame, mask= mask)

	# This is to draw the contours around the detected object
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	#### This is to filter out noise below a certain area####
	for contour in contours:
		# To find the laplacian  of the image to find the blur level and add it to the csv file
		img2 = result [:, :, 2]
		#cv2.imshow('keerat',img2)
		blur = variance_of_laplacian(img2)
		# This displays the contours
		cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	cv2.imshow('result',result)

	k = cv2.waitKey(40) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
