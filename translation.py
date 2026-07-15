import cv2 as cv
import numpy as np
img = cv.imread('dog.webp')
cv.imshow('Original', img)
cv.moveWindow('Original', 0, 0)

# def translation(img, x, y):
#     translate_matrix=np.float32([[1,0,x],[0,1,y]])
#     dimensions=(img.shape[1], img.shape[0])
#     return cv.warpAffine(img, translate_matrix, dimensions)


# translated = translation(img, -300, -300)
# cv.imshow('Translated', translated)
# cv.moveWindow('Translated', 400, 0)


# #rotation
# def rotation(img,angle,rotPoint=None):
#     (height,width) = img.shape[:2]
#     if rotPoint is None:
#         rotPoint = (width//2, height//2)
#     rotation_matrix = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
#     return cv.warpAffine(img, rotation_matrix, (width, height))


# rotated = rotation(img, -45)
# cv.imshow('Rotated', rotated)
# cv.moveWindow('Rotated', 800, 0)


#flipping
flip = cv.flip(img, 1)
cv.imshow('Flipped', flip)
#cropping
cropped = img[50:400, 100:400]
cv.imshow('Cropped', cropped)



cv.waitKey(0)
cv.destroyAllWindows()

