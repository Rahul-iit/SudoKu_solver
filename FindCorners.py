import cv2
import numpy as np
import operator

def preProcess(im):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 3, 2)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated = cv2.dilate(adaptive_thresh, kernel, iterations = 1)
    return dilated
    
def findCorners(im):
    contours, _ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]  
    bottom_right, temp = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, temp     = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, temp  = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, temp    = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [tuple(polygon[top_left][0]), tuple(polygon[top_right][0]), tuple(polygon[bottom_right][0]), tuple(polygon[bottom_left][0])]

'''
image = cv2.imread('image.jpg')
pre_processed = preProcess(image)
corners = findCorners(pre_processed)

im_corners = image.copy()
for corner in corners:
    cv2.circle(im_corners, corner, 5, (0,0,255), -1)
cv2.imshow('All Corners', im_corners)
cv2.waitKey(0)  
cv2.destroyAllWindows()

'''