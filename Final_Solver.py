import cv2
import numpy as np
import FindCorners
import Crop
import Grid
import operator
import math
import ExtractCells
import ReadSudoKu
import solver
from keras.models import load_model

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

sudoku = ReadSudoKu.readSudoku(model, extractedCells)
print("Unsolved sudoku :")
PRINT(sudoku)

solver.solve_sudoku(sudoku)
print("Solved sudoku :")
PRINT(sudoku)

