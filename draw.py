import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')

# blank[200:300, 300:400] = (0,255,0)

# cv.imshow('Green', blank)



#rectangled
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0), thickness=-1)
cv.imshow('Rectangle', blank)

#circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1)
cv.imshow('Circle', blank)

#line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,0,0), thickness=2)
cv.imshow('Line', blank)


#text
cv.putText(blank,'Hello My name is Kanth',(10,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), thickness=2)
cv.imshow('Text', blank)


cv.waitKey(0)
cv.destroyAllWindows()

