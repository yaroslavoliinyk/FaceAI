from PIL import Image
import os
import glob
import cv2
import numpy as np


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and not file.startswith('.'):
            yield file


def descale_imgs_coef_many(coef):
    img_names = listdir_nohidden("imgs/experiment/")
    
    path = "imgs/descaled_"+str(coef)
    for img_name in img_names:
        img = Image.open(img_name)
        width, height = img.size
        new_width     = int(width * coef)
        new_height    = int(height * coef)
        new_size      = (new_width, new_height)
        new_img       = img.resize(new_size)
        new_img_name  = img_name[:-5] + "_" + img_name[-5:]
        new_img.save(new_img_name)


def print_image(image_name, image):
    width, height = image.size
    varience = get_varience(image_name)
    sharpness = get_sharpness_brenner(image_name)
    pixel_num = width * height
    print("Image: " + image_name)
    print("width %d, height %d", width, height)
    print("Varience normalized: " + str(varience/pixel_num))
    print("Sharpness normalized: " + str(sharpness/pixel_num))


def show_varience_sharpness():
    img_names = files("imgs/experiment/")
    
    for img_name in img_names:
        img_name = "imgs/experiment/" + img_name
        img = Image.open(img_name)
        name = img_name.split('/')[-1]
        img_name_descale = "imgs/experiment/desc/" + name
        img_descale = Image.open(img_name_descale)
        # Upscaled
        print_image(img_name, img)
        print()
        #Descaled
        print_image(img_name_descale, img_descale)
        print("---------------------------------------------")
        print()
        print()


show_varience_sharpness()



