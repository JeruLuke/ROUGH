import cv2 
import numpy as np 

'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import numpy as np
%matplotlib inline
import matplotlib.image as mpimg


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation
    '''
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img
    
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
plt.figure(figsize=(12,12))
for i in range(100):
    ax1 = plt.subplot(gs1[i])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    img = transform_image(image,20,10,5,brightness=1)

    plt.subplot(10,10,i+1)
    plt.imshow(img)
    plt.axis('off')

plt.show()

'''
    
path = 'C:/Users/selwyn77/Desktop/plant/charlock/0a7e1ca41.png'

img = cv2.imread(path , 1)
cv2.imshow('original', img)

def translate(img, M):
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted

def shear(img, shear_range):
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range/2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, shear_M, (img.shape[1], img.shape[0]))

def brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    print random_bright
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
        
    
M = np.float32([[1,0,20],[0,1,40]])  
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0])) # we get shifted image 
cv2.imshow("Shifted", shifted) 


M = np.float32([[1,0,0],[0,1,-60]]) 
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0])) 
cv2.imshow("shifted to up", shifted) 

cv2.imshow('translate_1', translate(img, np.float32([[1,0,20],[0,1,40]])))
cv2.imshow('translate_2', translate(img, np.float32([[1,0,0],[0,1,-60]])))


cv2.imshow('shear_1', shear(img, 1))
cv2.imshow('shear_2', shear(img, 5))
cv2.imshow('shear_3', shear(img, 10))
cv2.imshow('shear_4', shear(img, 15))
cv2.imshow('shear_5', shear(img, 20))

cv2.imshow('brigh_1', brightness(img))
cv2.imshow('brigh_2', brightness(img))
cv2.imshow('brigh_3', brightness(img))
cv2.imshow('brigh_4', brightness(img))
cv2.imshow('brigh_5', brightness(img))
cv2.imshow('brigh_6', brightness(img))

cv2.waitKey(0) 
cv2.destroyAllWindows()



