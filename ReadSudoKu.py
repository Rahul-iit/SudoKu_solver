import cv2
import numpy as np
import FindCorners
import Crop
import Grid
import operator
import math
import ExtractCells
from keras.models import load_model


def readSudoku(model, cells):
    sudoku_matrix = np.full((1, 81), -1)

    for i, cell in enumerate(cells):
        if(np.allclose(cell, 0)):
            sudoku_matrix[0][i] = 0
        else:
            kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
            res = cv2.erode(cell.copy(), kernel, iterations=1)
            res = cv2.bitwise_not(res)
            dim = (28, 28)
            res = cv2.resize(res, dim, interpolation=cv2.INTER_AREA)
            res = np.reshape(res, (1, 28, 28, 1))

            res = res.astype(dtype="float32")
            if(res.max() != 0.):
                res /= 255
            nb = model.predict(res)

            for j in range(nb.shape[0]):
                for k in range(nb.shape[1]):
                    if(nb[j][k] > 0.9):
                        sudoku_matrix[0][i] = int(k)

    sudoku_matrix = sudoku_matrix.reshape((9, 9))
    return sudoku_matrix.T

def PRINT(grid):
	for i in range(1,10):
		for j in range(1,10):
			print(grid[i-1][j-1],end = ' ')
			if(j%3 == 0):
				print('| ',end = ' ')
		
		if(i%3 == 0):
			print("\n-------------------------")
		else:
			print()


'''
model = load_model('cnn_model.h5')
image = cv2.imread('image.jpeg')
pre_processed = FindCorners.preProcess(image)
corners = FindCorners.findCorners(pre_processed)
crop_img = Crop.crop(pre_processed, corners)

squares = Grid.grid(crop_img)

squares_int = [[tuple(int(x) for x in tup) for tup in square] for square in squares]

im_squares = cv2.cvtColor(crop_img.copy(), cv2.COLOR_GRAY2RGB)
for square in squares_int:
    cv2.rectangle(im_squares,square[0],square[1],(0,255,0),1)
cells = ExtractCells.extractCells(crop_img, squares_int)

extractedCells = []
for cell in cells:
    h, w = cell.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = ExtractCells.LargestConnectedComponent(cell, [margin, margin], [w - margin, h - margin])
    extractedCells.append(ExtractCells.extractDigit(cell, bbox, 28))

columns = []
with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 255) for img in extractedCells]
for i in range(9):
    column = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=0)
    columns.append(column)
Ext_cell = np.concatenate(columns, axis=1)

sudoku = readSudoku(model, extractedCells)
print("Extracted sudoku :")
PRINT(sudoku)
'''

