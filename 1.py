import cv2
import numpy as np

#imagePath = 'C:/Users/524316/Desktop/Ratan Presentation/4.jpg'
image = cv2.imread('1.jpg')
#image = cv2.resize(image, (0,0), fx=0.35, fy=0.35) 
image = cv2.resize(image, (0,0), fx=1.35, fy=1.35) 
cv2.imshow("Original", image)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl1 = clahe.apply(image[:,:,0])
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl2 = clahe.apply(image[:,:,1])
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl3 = clahe.apply(image[:,:,2])

limg = cv2.merge((cl1, cl2, cl3))
cv2.imwrite('limg1.jpg', limg)


equb = cv2.equalizeHist(image[:,:,0])
equg = cv2.equalizeHist(image[:,:,1])
equr = cv2.equalizeHist(image[:,:,2])
equimg = cv2.merge((equb, equg, equr))
cv2.imshow('equimg', equimg)
#cv2.imshow("Originalb.jpg", image[:,:,0])
#cv2.imshow("Originalg.jpg", image[:,:,1])
#cv2.imshow("Originalr.jpg", image[:,:,2])


hsv = cv2.cvtColor(limg, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv_h.jpg", hsv[:,:,0])
cv2.imshow("hsv_s.jpg", hsv[:,:,1])
cv2.imshow("hsv_v.jpg", hsv[:,:,2])

lab = cv2.cvtColor(limg, cv2.COLOR_BGR2LAB)
cv2.imshow("lab.jpg", lab)
cv2.imshow("lab_l.jpg", lab[:,:,0])
cv2.imshow("lab_a.jpg", lab[:,:,1])
cv2.imshow("lab_b.jpg", lab[:,:,2])


hls = cv2.cvtColor(limg, cv2.COLOR_BGR2HLS)
cv2.imshow("hls_h.jpg", hls[:,:,0])
cv2.imshow("hls_l.jpg", hls[:,:,1])
cv2.imshow("hls_s.jpg", hls[:,:,2])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mean = np.mean(gray)

#ret,thresh1 = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

#th3 = cv2.adaptiveThreshold(hsv[:,:,2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
#cv2.imshow("th3.jpg", th3)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(hsv[:,:,2])
cv2.imshow('CLAHE output', cl)

equ = cv2.equalizeHist(hsv[:,:,2])
cv2.imshow("equ.jpg", equ)


cv2.waitKey(0)
cv2.destroyAllWindows()