import cv2 as cv
import numpy as np

img = cv.imread("IMG_0327.jpg")

a = [3024, 4032]


poly = np.array([[0.27, 0.77], [0.27, 1], [1, 0.77], [1, 1]], np.int32)
poly = np.array(
    [
        [830, 3120],
        [3024, 3120],
        [3024, 4032],
        [830, 4032],
    ]
)

img = cv.fillPoly(
    img,
    pts=[poly],
    color=(255, 0, 0),
)

cv.imshow("Image", img)
cv.waitKey(0)
