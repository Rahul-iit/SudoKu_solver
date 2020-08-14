import cv2
import numpy as np
import FindCorners
import Crop
import Grid
import operator
import math


def extractCells(im, squares):
    cells = []
    for i, square in enumerate(squares):
        cell = im[square[0][1]:square[1][1], square[0][0]:square[1][0]]
        cells.append(cell)
    return cells

def LargestConnectedComponent(inp_img, scan_tl=None, scan_br=None):
    im = inp_img.copy() 
    height, width = im.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if(im.item(y, x) == 255 and x < width and y < height):  
                area = cv2.floodFill(im, None, (x, y), 64)
                if(area[0] > max_area):  
                    max_area = area[0]
                    seed_point = (x, y)

    for x in range(width):
        for y in range(height):
            if im.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(im, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8) 

    if all([p is not None for p in seed_point]):
        cv2.floodFill(im, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if im.item(y, x) == 64:  
                cv2.floodFill(im, mask, (x, y), 0)

            if im.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return im, np.array(bbox, dtype='float32'), seed_point


def extractDigit(cell, bbox, size):
    cell = cell[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    if w > 0 and h > 0 and (w * h) > 100 and cell.size:
        return scale_and_centre(cell, size, 4)
    return np.zeros((size, size), np.uint8)




def scale_and_centre(img, size, margin=0, background=0):
	h, w = img.shape[:2]

	def centre_pad(length):
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))



'''
image = cv2.imread('image.jpeg')
pre_processed = FindCorners.preProcess(image)
corners = FindCorners.findCorners(pre_processed)
crop_img = Crop.crop(pre_processed, corners)

squares = Grid.grid(crop_img)

squares_int = [[tuple(int(x) for x in tup) for tup in square] for square in squares]

im_squares = cv2.cvtColor(crop_img.copy(), cv2.COLOR_GRAY2RGB)
for square in squares_int:
    cv2.rectangle(im_squares,square[0],square[1],(0,255,0),1)
cells = extractCells(crop_img, squares_int)

extractedCells = []
for cell in cells:
    h, w = cell.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = LargestConnectedComponent(cell, [margin, margin], [w - margin, h - margin])
    extractedCells.append(extractDigit(cell, bbox, 28))

columns = []
with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 255) for img in extractedCells]
for i in range(9):
    column = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=0)
    columns.append(column)
Ext_cell = np.concatenate(columns, axis=1)

cv2.imshow("Extracted Cells", Ext_cell)
cv2.waitKey(0)  
cv2.destroyAllWindows()
'''
