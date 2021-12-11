from PIL import Image
from other.prg.ISR.models import RDN
from other.prg.ISR.models import RRDN
from other.prg.ISR.models import Discriminator
from other.prg.ISR.models import Cut_VGG19
from other.prg.ISR.train import Trainer
from modules.CNNIQAnet import CNNIQAnet
from other.image_assessment_CNNIQA.IQADataset import NonOverlappingCropPatches
from skimage import filters

import re
import torch
import numpy as np
import os
import cv2
import math

class AppImage():

    #APP_IMAGE_FOLDER           = "app_size"
    APP_IMAGE_UPSCALE_PREDICAT = "upscale_"
    ALGORITHMS                 = ["noise-cancel", "psnr-small", "psnr-large"]
    MODEL                      = "noise-cancel"
    INITIAL_IMAGE_PATH         = "imgs/empty-avatar.png"
    # app_image_width = 300, app_image_height = 300
    def __init__(self, image_path):
        if(image_path == None):
            image_path = self.INITIAL_IMAGE_PATH
        
        self._image_path                       = image_path
        img = Image.open(self._image_path)
        width, height                          = img.size
        self._app_image_width                  = width
        self._app_image_height                 = height


    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, path):
        self._image_path = path

    @property
    def app_image_width(self):
        img = Image.open(self._image_path)
        width, height                          = img.size
        self._app_image_width                  = width
        return self._app_image_width

    @property
    def app_image_height(self):
        img = Image.open(self._image_path)
        width, height                          = img.size
        self._app_image_height                 = height
        return self._app_image_height


    def get_energy(self):
        #original image
        img = cv2.imread(self.image_path)
        # gray color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        shape = np.shape(img)
        out = 0
        for y in range(0, shape[1]-1):
            for x in range(0, shape[0]-1):
                out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
        
        img = Image.open(self.image_path)
        width, height = img.size
        out /= width * height * width * height

        return round(out, 3) 


    def get_sharpness_brenner(self):
        '''
        :param img:narray             the clearer the image,the larger the return value
        :return: float 
        '''
        #original image
        img = cv2.imread(self.image_path)
        shape = np.shape(img)
        # gray color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = 0
        for y in range(0, shape[1]):
            for x in range(0, shape[0]-2):
                
                out+=(int(img[x+2,y])-int(img[x,y]))**2
                
        img = Image.open(self.image_path)
        width, height = img.size
        out /= width * height

        return round(out, 3) 


    def get_SMD(self):
        #original image
        img = cv2.imread(self.image_path)
        # gray color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for y in range(0, shape[1]-1):
            for x in range(0, shape[0]-1):
                out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
                out+=math.fabs(int(img[x,y]-int(img[x+1,y])))

        img = Image.open(self.image_path)
        width, height = img.size
        out /= width * height

        return round(out, 3)


    def get_histogram(self):
        src = cv2.imread(cv2.samples.findFile(self.image_path))
        if src is None:
            print('Could not open or find the image:', img_path)
            exit(0)
        bgr_planes = cv2.split(src)
        histSize = 256
        histRange = (0, 256) # the upper boundary is exclusive
        accumulate = False
        b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
        hist_w = 512
        hist_h = 400
        bin_w = int(round( hist_w/histSize ))
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        for i in range(1, histSize):
            cv2.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                    ( bin_w*(i), hist_h - int(b_hist[i]) ),
                    ( 255, 0, 0), thickness=2)
            cv2.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                    ( bin_w*(i), hist_h - int(g_hist[i]) ),
                    ( 0, 255, 0), thickness=2)
            cv2.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                    ( bin_w*(i), hist_h - int(r_hist[i]) ),
                    ( 0, 0, 255), thickness=2)
        return histImage
        

    def _get_image_name(self, image_path):
        return image_path.split("/")[-1]


    def _get_image_folder(self, image_path):
        return image_path.split(self._get_image_name(image_path))[0]

    
class OriginalAppImage(AppImage):
    INITIAL_IMAGE_PATH         = "imgs/empty-avatar.png"

    def __init__(self, image_path=None):
        super().__init__(image_path)
        self._upscaled_app_image = UpscaledAppImage()
        self.is_upscaled        = False


    @property
    def upscaled_app_image(self):
        return self._upscaled_app_image


    def upscale(self, algorithm_name):
        image_name   = self._get_image_name(self.image_path)
        image_folder = self._get_image_folder(self.image_path)
        result_image = self.__run_isr(algorithm_name, self.image_path, image_folder, image_name)
        upscaled_image_path = image_folder + "/" + self.APP_IMAGE_UPSCALE_PREDICAT + image_name
        
        self._upscaled_app_image.image_path = upscaled_image_path
        upscaled_image_path.encode('unicode_escape')
        result_image.save(upscaled_image_path)
        self.is_upscaled = True

    # available models: psnr-large, psnr-small, noise-cancel
    def __run_isr(self, model, image_path, image_folder, image_name):
        img     = Image.open(image_path)
        lr_img  = np.array(img)
        rdn     = RDN(weights=model)
        sr_img  = rdn.predict(lr_img)
        result_img = Image.fromarray(sr_img)
        return result_img


    def is_upscaled(self):
        return self.is_upscaled



class UpscaledAppImage(AppImage):
    
    def __init__(self, image_path=None):
        super().__init__(image_path)

