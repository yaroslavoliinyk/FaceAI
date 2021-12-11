import cv2
import numpy as np
import math

from PIL import Image

#
def entropy(image_path):
    #original image
    img = cv2.imread(image_path)
    # gray color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    [rows, cols] = img.shape
    h = 0
    hist_gray = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
    # hn valueis not correct
    hb = np.zeros((256, 1), np.float32)
    #hn = np.zeros((256, 1), np.float32)
    for j in range(0, 256):
        hb[j, 0] = hist_gray[j, 0] / (rows*cols)
    for i in range(0, 256):
        if hb[i, 0] > 0:
            h = h - (hb[i, 0])*math.log(hb[i, 0],2)
                
    out = h
    return out

#
def vollath(image_path):
    #original image
    img = cv2.imread(image_path)
    # gray color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for y in range(0, shape[1]):
        for x in range(0, shape[0]-1):
            out+=int(img[x,y])*int(img[x+1,y])

    img = Image.open(image_path)
    width, height = img.size
    out /= width * height

    return out

# - use
def energy(image_path):
    #original image
    img = cv2.imread(image_path)
    # gray color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    shape = np.shape(img)
    out = 0
    for y in range(0, shape[1]-1):
        for x in range(0, shape[0]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
    
    img = Image.open(image_path)
    width, height = img.size
    out /= width * height * width * height

    return out

#
def varience(image_path):
    #original image
    img = cv2.imread(image_path)
    # gray color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for y in range(0,shape[1]):
        for x in range(0,shape[0]):
            out+=(img[x,y]-u)**2
    
    img = Image.open(image_path)
    width, height = img.size
    out /= width * height

    return out

# - use
def SMD(image_path):
    #original image
    img = cv2.imread(image_path)
    # gray color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    shape = np.shape(img)
    out = 0
    for y in range(0, shape[1]-1):
        for x in range(0, shape[0]-1):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))

    img = Image.open(image_path)
    width, height = img.size
    out /= width * height

    return out


def SMD2(image_path):
    #original image
    img = cv2.imread(image_path)    
    # gray color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    shape = np.shape(img)
    out = 0
    for y in range(0, shape[1]-1):
        for x in range(0, shape[0]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))

    img = Image.open(image_path)
    width, height = img.size
    out /= width * height

    return out

# - use
def brenner(image_path):
    #original image
    img = cv2.imread(image_path)
    shape = np.shape(img)
    # gray color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    out = 0
    for y in range(0, shape[1]):
        for x in range(0, shape[0]-2):
            
            out+=(int(img[x+2,y])-int(img[x,y]))**2

    img = Image.open(image_path)
    width, height = img.size
    out /= width * height
            
    return out
