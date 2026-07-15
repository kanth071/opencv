import cv2 as cv
img = cv.imread('dog.webp')

if img is None:
    print("ERROR: Image not loaded!")
else:
    cv.imshow('dog.webp', img)
    
    # Gray scaling
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)


    
    #Blur 
    blur = cv.GaussianBlur(img, (7,7), 0)
    cv.imshow('Blur', blur)
    
    #Resizing
    resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
    cv.imshow('Resized', resized)

    #edge cascade
    canny = cv.Canny(resized, 125, 175)
    cv.imshow('Canny Edges', canny)

    #dilating
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    dilated = cv.dilate(canny, kernel, iterations=3)
    cv.imshow('Dilated', dilated)


    # Crop from resized image for better visibility
    cropped = resized[50:300, 50:300]
    cv.imshow('Cropped',cropped)
    
    # Enhance cropped image for better clarity
    cropped_gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    cropped_canny = cv.Canny(cropped_gray, 100, 150)
    cv.imshow('Cropped Edges', cropped_canny)

cv.waitKey(0)
cv.destroyAllWindows()


