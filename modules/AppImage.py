from PIL import Image
from other.prg.ISR.models import RDN
from other.prg.ISR.models import RRDN
from other.prg.ISR.models import Discriminator
from other.prg.ISR.models import Cut_VGG19
from other.prg.ISR.train import Trainer
from modules.CNNIQAnet import CNNIQAnet
from other.image_assessment_CNNIQA.IQADataset import NonOverlappingCropPatches

import re
import torch
import numpy as np
import os

class AppImage():

    APP_IMAGE_FOLDER           = "app_size"
    APP_IMAGE_UPSCALE_PREDICAT = "upscale_"
    INITIAL_IMAGE_NAME         = "imgs/empty-avatar.png"
    IMAGE_DOWNSCALE_PREDICAT   = "downscale_"
    ALGORITHMS                 = ["noise-cancel", "psnr-small", "psnr-small"]
    MODEL                      = "noise-cancel"

    # app_image_width = 300, app_image_height = 300
    def __init__(self, image_path):
        self.image_path         = image_path
        img = Image.open(self.image_path)
        width, height = img.size
        self.app_image_width                  = width
        self.app_image_height                 = height
        self.on_screen_descaled               = False
        self.upscaled_image_path              = "imgs/empty-avatar.png"
        # Image after descaling
        self.descaled_image_path              = "imgs/empty-avatar.png"
        self.upscaled_descaled_image_path     = "imgs/empty-avatar.png"


    # Here can be either -5% image or original image(depends on path it's located in)
    # Bool type to know for sure if the image we give was indeeed descaled on 5% or it's an original 
    def upscale(self):
        if(self.get_on_screen_descaled()):
            image_name   = self.__get_image_name(self.descaled_image_path)
            image_folder = self.__get_image_folder(self.descaled_image_path)
            self.upscaled_descaled_image_path     = self.__run_isr_save_upscaled_image(self.MODEL, self.descaled_image_path, image_folder, image_name)
            self.upscaled_descaled_image_app_path = self.__resize_and_save_image_path(self.upscaled_descaled_image_path, self.app_image_width, self.app_image_height)
        else:
            image_name   = self.__get_image_name(self.image_path)
            image_folder = self.__get_image_folder(self.image_path)
            self.upscaled_image_path = self.__run_isr_save_upscaled_image(self.MODEL, self.image_path, image_folder, image_name)
            self.upscaled_image_app_path = self.__resize_and_save_image_path(self.upscaled_image_path, self.app_image_width, self.app_image_height) 
        

    def downscale_minus_5(self):
        if(self.descaled_image_path == self.INITIAL_IMAGE_NAME):
            image_name   = self.__get_image_name(self.image_path)
            image_folder = self.__get_image_folder(self.image_path)
            coef         = 0.95
            self.descaled_image_path         = self.__descale_imgs_coef(self.image_path, image_folder, image_name, coef)
            self.descaled_image_app_path     = self.__resize_and_save_image_path(self.descaled_image_path, self.app_image_width, self.app_image_height) 
        self.set_on_screen_descaled(True)


    def is_upscaled(self):
        if (self.upscaled_image_path == self.INITIAL_IMAGE_NAME):
            return False
        else:
            return True

   
    def set_on_screen_descaled(self, is_descaled):
        self.on_screen_descaled = is_descaled


    def get_on_screen_descaled(self):
        return self.on_screen_descaled


    def get_image_path(self):
        return self.image_path


    def get_upscaled_image_path(self):
        if(self.is_upscaled()):
            return self.upscaled_image_path
        else:
            return None


    def get_app_image_width(self):
        return self.app_image_width


    def get_app_image_height(self):
        return self.app_image_height


    # Images for showing in the app
    def get_image_app_path(self):
        return self.image_app_path


    def get_upscaled_image_app_path(self):
        return self.upscaled_image_app_path


    def get_descaled_image_app_path(self):
        return self.descaled_image_app_path


    def get_upscaled_descaled_image_app_path(self):
        return self.upscaled_descaled_image_app_path


    # Real images
    def get_image_path(self):
        return self.image_path


    def get_upscaled_image_path(self):
        return self.upscaled_image_path


    def get_descaled_image_path(self):
        return self.descaled_image_path


    def get_upscaled_descaled_image_path(self):
        return self.upscaled_descaled_image_path


    def assess_image(self, image_path):
        if image_path == self.INITIAL_IMAGE_NAME:
            return "?.??"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNIQAnet(ker_size=7,
                        n_kers=50,
                        n1_nodes=800,
                        n2_nodes=800).to(device)

        model.load_state_dict(torch.load("other/image_assessment_CNNIQA/models/CNNIQA-LIVE"))

        im = Image.open(image_path).convert('L')
        patches = NonOverlappingCropPatches(im, 32, 32)

        model.eval()
        with torch.no_grad():
            patch_scores = model(torch.stack(patches).to(device))
            score        = str(50. - model(torch.stack(patches).to(device)).mean())
            score  = re.findall("\d+\.\d+", score)[0]
            return score

    
    def __get_image_name(self, image_path):
        return image_path.split("/")[-1]


    def __get_image_folder(self, image_path):
        return image_path.split(self.__get_image_name(image_path))[0]


    def __descale_imgs_coef(self, image_path, image_folder, image_name,  coef):
        img = Image.open(image_path)
        width, height = img.size
        new_width     = int(width * coef)
        new_height    = int(height * coef)
        new_size      = (new_width, new_height)
        new_img       = img.resize(new_size)

        descaled_image_path = image_folder + "/" + self.IMAGE_DOWNSCALE_PREDICAT + image_name
        descaled_image_path.encode('unicode_escape')
        new_img.save(descaled_image_path)
        return descaled_image_path


    def __resize_and_save_image_path(self, image_path, app_image_width, app_image_height):
        image_folder = self.__get_image_folder(image_path)
        image_name   = self.__get_image_name(image_path)
        img = Image.open(image_path)
        width, height = img.size

        bigger_image_side_length = None
        smaller_app_side_lenght = None

        if(width > height):
            bigger_image_side_length = width
        else:
            bigger_image_side_length = height

        if(app_image_width < app_image_height):
            smaller_app_side_lenght = app_image_width
        else:
            smaller_app_side_lenght = app_image_height

        coef          = smaller_app_side_lenght / bigger_image_side_length
        new_width     = int(width * coef)
        new_height    = int(height * coef)
        new_size      = (new_width, new_height)
        app_img       = img.resize(new_size)
        

        image_app_folder = image_folder + self.APP_IMAGE_FOLDER
        image_app_path   = image_app_folder + "/" + image_name
        image_app_folder.encode('unicode_escape')
        image_app_path.encode('unicode_escape')
        #if any(File.endswith(self.APP_IMAGE_FOLDER) for File in os.listdir(".")):
        if not os.path.exists(image_app_folder):
            os.mkdir(image_app_folder)
        print(image_app_folder)
        print(image_app_path)
        app_img.save(image_app_path)

        return image_app_path


    # available models: psnr-large, psnr-small, noise-cancel
    def __run_isr_save_upscaled_image(self, model, image_path, image_folder, image_name):
        img     = Image.open(image_path)
        lr_img  = np.array(img)
        rdn     = RDN(weights=model)
        sr_img  = rdn.predict(lr_img)
        self.result_img = Image.fromarray(sr_img)
        
        upscaled_image_path = image_folder + "/" + self.APP_IMAGE_UPSCALE_PREDICAT + image_name
        upscaled_image_path.encode('unicode_escape')
        self.result_img.save(upscaled_image_path)
        
        return upscaled_image_path



class OriginaAppImage(AppImage):
    
    def __init__():
        pass

class UpscaledAppImage(AppImage):
    
    def __init__():
        pass