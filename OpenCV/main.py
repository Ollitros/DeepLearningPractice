import numpy as np
import cv2 as cv

img = cv.imread('data/hatiko.jpg', 0)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('data/hatiko_gray.jpg', img)