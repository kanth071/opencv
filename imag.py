import cv2 as cv
# img = cv.imread('Screenshot 2026-06-17 125148.png')
# cv.imshow('Screenshot 2026-06-17 125148', img)

# cv.waitKey(0)
import cv2 as cv
capture = cv.VideoCapture('dog.mp4.mp4')
while True:
    istrue,frame =capture.read()
    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()