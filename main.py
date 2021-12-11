import numpy as np
import os

from PIL import Image
from other.prg.ISR.models import RDN
from other.prg.ISR.models import RRDN
from other.prg.ISR.models import Discriminator
from other.prg.ISR.models import Cut_VGG19
from other.prg.ISR.train import Trainer


from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from image_assessment_CNNIQA.IQADataset import NonOverlappingCropPatches

"""

        Class, where all the experiments take place.
In here the author descales images to 95%, 90%, 85%, 75%,
and a half of quality. The results are shown in the corresponding
folders(See the code).

"""


def descale_imgs():
    img_names = os.listdir("imgs/originals/")
    for img_name in img_names:
        img = Image.open("imgs/originals/" + img_name)
        width, height = img.size
        new_width = width // 2
        new_height = height // 2
        new_size = (new_width, new_height)
        new_img = img.resize(new_size)
        new_img.save("imgs/descaled/" + img_name)


def descale_imgs_coef(coef):
    img_names = os.listdir("imgs/originals/")
    path = "imgs/descaled_" + str(coef)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    for img_name in img_names:
        img = Image.open("imgs/originals/" + img_name)
        width, height = img.size
        new_width = int(width * coef)
        new_height = int(height * coef)
        new_size = (new_width, new_height)
        new_img = img.resize(new_size)
        new_img.save("imgs/descaled_" + str(coef) + "/" + img_name)


def isr_upscale(img_path, folder_name):
    try:
        os.mkdir("imgs/" + folder_name)
    except OSError:
        print("Creation of the directory %s failed" % "imgs/" + folder_name)
    else:
        print("Successfully created the directory %s " % "imgs/" + folder_name)
    img = Image.open(img_path)
    lr_img = np.array(img)
    rdn = RDN(weights="noise-cancel")
    sr_img = rdn.predict(lr_img)
    result_image = Image.fromarray(sr_img)
    result_image.save("imgs/" + folder_name + "/" + "upscaled_after_descaling.jpeg")


# available models: psnr-large, psnr-small, noise-cancel
def run_isr(model):
    # img_names = os.listdir("imgs/descaled/image-3.jpeg")
    img = Image.open("imgs/descaled/scholar.jpg")
    lr_img = np.array(img)
    rdn = RDN(weights=model)
    sr_img = rdn.predict(lr_img)
    res_img = Image.fromarray(sr_img)
    res_img.save("imgs/isr_psnr-" + model + "/" + "scholar.jpg")


def train_models():
    # Create the models
    lr_train_patch_size = 40
    layers_to_extract = [5, 9]
    scale = 2
    hr_train_patch_size = lr_train_patch_size * scale

    rrdn = RRDN(
        arch_params={"C": 4, "D": 3, "G": 64, "G0": 64, "T": 10, "x": scale},
        patch_size=lr_train_patch_size,
    )
    f_ext = Cut_VGG19(
        patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract
    )
    discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)
    # Create the trainer
    loss_weights = {
        "generator": 0.0,
        "feature_extractor": 0.0833,
        "discriminator": 0.01,
    }
    losses = {
        "generator": "mae",
        "feature_extractor": "mse",
        "discriminator": "binary_crossentropy",
    }

    log_dirs = {"logs": "./logs", "weights": "./weights"}

    learning_rate = {
        "initial_value": 0.0004,
        "decay_factor": 0.5,
        "decay_frequency": 30,
    }

    flatness = {"min": 0.0, "max": 0.15, "increase": 0.01, "increase_frequency": 5}

    trainer = Trainer(
        generator=rrdn,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir="low_res/training/images",
        hr_train_dir="high_res/training/images",
        lr_valid_dir="low_res/validation/images",
        hr_valid_dir="high_res/validation/images",
        loss_weights=loss_weights,
        learning_rate=learning_rate,
        flatness=flatness,
        dataname="image_dataset",
        log_dirs=log_dirs,
        weights_generator=None,
        weights_discriminator=None,
        # Number of validation images:
        # (originally was 40)
        n_validation=8,
    )
    # Training
    trainer.train(
        epochs=80,
        steps_per_epoch=500,
        batch_size=16,
        monitored_metrics={"val_PSNR_Y": "max"},
    )


class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #

        h = self.conv1(x)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  # max-min pooling
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)
        return q


"""
    ```bash
    python test_demo.py --im_path=data/I03_01_1.bmp
    ```
"""


def image_assessment(img_path):
    # parser = ArgumentParser(description='PyTorch CNNIQA test demo')
    # parser.add_argument(img_path)
    # parser.add_argument("--model_file", type=str, default='image_assessment_CNNIQA/models/CNNIQA-LIVE',
    #                    help="model file (default: image_assessment_CNNIQA/models/CNNIQA-LIVE)")

    # args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNIQAnet(ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800).to(device)

    model.load_state_dict(torch.load("image_assessment_CNNIQA/models/CNNIQA-LIVE"))

    im = Image.open(img_path).convert("L")
    patches = NonOverlappingCropPatches(im, 32, 32)

    model.eval()
    with torch.no_grad():
        patch_scores = model(torch.stack(patches).to(device))
        score = 50.0 - model(torch.stack(patches).to(device)).mean()
        print("score", score)


def main():
    # train_models()
    # run_isr("psnr-large")
    # run_isr("noise-cancel")
    image_assessment("imgs/descaled/image_2.jpeg")
    # run_isr("noise-cancel")
    # descale_imgs_coef(0.95)
    # descale_imgs_coef(0.9)
    # descale_imgs_coef(0.85)
    # descale_imgs_coef(0.75)
    # isr_upscale("imgs/descaled_0.95/image-3_descaled.jpeg", "upscaled_0.95")
    # isr_upscale("imgs/descaled_0.9/image-3_descaled.jpeg", "upscaled_0.9")
    # isr_upscale("imgs/descaled_0.85/image-3_descaled.jpeg", "upscaled_0.85")
    # isr_upscale("imgs/descaled_0.75/image-3_descaled.jpeg", "upscaled_0.75")
    """img = Image.open("imgs/descaled_0.75/image-3.jpeg")
    width, height = img.size
    new_width     = width // 2
    new_height    = height // 2
    new_size      = (new_width, new_height)
    new_img       = img.resize(new_size)
    new_img.save("imgs/descaled_0.75/image-3_descaled.jpeg")"""


if __name__ == "__main__":
    main()


"""im = Image.open("imgs/me.jpg")

width, height = im.size
new_width     = width // 2
new_height    = height // 2
new_size      = (new_width, new_height)

new_img       = im.resize(new_size)
new_img.save("imgs/me_descaling.jpg")"""

