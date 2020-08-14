import cv2
import numpy as np
import math
import FindCorners

def distance(point0, point1):
    return math.sqrt((point0[0] - point1[0])**2 + (point0[1] - point1[1])**2)

def crop(imgage, corners):
	src = np.array([corners[0], corners[1], corners[2], corners[3]], dtype='float32')
	side = max([
            distance(corners[2], corners[1]),
            distance(corners[0], corners[3]),
            distance(corners[2], corners[3]),
            distance(corners[0], corners[1])
	    ])
	dist = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32') 

	m = cv2.getPerspectiveTransform(src, dist)
	return cv2.warpPerspective(imgage, m, (int(side), int(side)))

'''
image = cv2.imread('image.jpeg')
pre_processed = FindCorners.preProcess(image)
corners = FindCorners.findCorners(pre_processed)

Crop_img = crop(pre_processed, corners)
cv2.imshow('After cropping', Crop_img)
cv2.waitKey(0)  
cv2.destroyAllWindows()
'''