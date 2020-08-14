import cv2
import numpy as np
import FindCorners
import Crop

def grid(im, size=9):
    squares = []
    side = im.shape[:1][0]
    side = side / size
    for i in range(9):
        for j in range(9):
            p1 = (i * side, j * side)
            p2 = ((i + 1) * side, (j + 1) * side) 
            squares.append((p1, p2))
    return squares

'''
image = cv2.imread('image.jpeg')
pre_processed = FindCorners.preProcess(image)
corners = FindCorners.findCorners(pre_processed)
crop_img = Crop.crop(pre_processed, corners)

squares = grid(crop_img)

# Convert to int
squares_int = [[tuple(int(x) for x in tup) for tup in square] for square in squares]

im_squares = cv2.cvtColor(crop_img.copy(), cv2.COLOR_GRAY2RGB)
for square in squares_int:
    cv2.rectangle(im_squares,square[0],square[1],(0,255,0),1)

cv2.imshow('Grid applied', im_squares)
cv2.waitKey(0)  
cv2.destroyAllWindows()

'''