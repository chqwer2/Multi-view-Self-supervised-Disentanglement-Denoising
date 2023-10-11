from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random
from torchvision import transforms

from options import utils_option
import os
import math
from lib import utils_image as util



class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()


    def get_parms(self):

        # ------------------------------------
        # Probability
        # ------------------------------------

        self.noise_p = self.opt["corruption"]["noise"]
        self.rain_p = self.opt["corruption"]["rain"]
        self.blur_p = self.opt["corruption"]["blur"]

        self.impainting_p = self.opt["corruption"]["impainting"]
        self.impainting_rate = self.opt["corruption"]["impainting_rate"]
        self.superresolution_p = self.opt["corruption"]["superresolution"]
        self.superresolution_scale = self.opt["corruption"]["superresolution_scale"]

        self.scale = self.opt["upscale"]
        self.L_size = self.image_size // self.scale

        self.sigma = self.datasets_opt['sigma'] if self.datasets_opt['sigma'] else 25
        self.sigma_test = self.datasets_opt['sigma_test'] if self.datasets_opt['sigma_test'] else self.sigma

        self.noise_type = self.opt["corruption"]["noise_type"]
        self.noise_num = self.opt["corruption"]["noise_num"]

        self.resize_size = self.image_size * 8

        self.DEBUG = False


    def combine_datasets_path(self):
        get_dataset = self.datasets_opt[self.dataset_type + "_dataset"]  # dataset list

        self.paths_H = []
        self.paths_L = []

        if isinstance(get_dataset, list):
            for dataset_name in get_dataset:
                self.get_datasets_path(dataset_name)
        else:
            self.get_datasets_path(get_dataset)

    def get_datasets_path(self, get_dataset):

        self.dataset_file = utils_option.json_parse(self.opt["dataset_file"])[get_dataset]
        self.shift = self.datasets_opt["shift"]

        self.n_channels = self.dataset_file['n_channels'] if self.dataset_file['n_channels'] else 3
        self.image_size = self.datasets_opt['H_size'] if self.datasets_opt['H_size'] else 64
        self.window_size = self.opt["netG"]["window_size"]

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H_filename = self.dataset_file['dataroot_H_filename']
        self.paths_L_filename = self.dataset_file['dataroot_L_filename']

        self.paths_H.extend(
            util.get_image_paths(os.path.join(self.dataset_file['dataroot_H'], self.paths_H_filename[0])))
        self.paths_H.sort()

        if self.dataset_file['dataroot_L'] != None:
            self.paths_L.extend(
                util.get_image_paths(os.path.join(self.dataset_file['dataroot_L'], self.paths_L_filename[0]))
            )
        else:
            self.paths_L.extend(self.paths_H)

        self.paths_L.sort()

        if self.DEBUG:
            for i in zip(self.paths_L, self.paths_H):
                print("data list:", i)



    def get_image_from_path(self, path):
        img = Image.open(path)
        if self.n_channels < 3:
            img = img.convert("L")
        img = np.asarray(img)
        return img

    def get_img_by_index(self, index):
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]

        img_H = self.get_image_from_path(H_path)

        if self.opt["task"] == "denoising" or self.opt["task"] == "med":
            img_H = self.albumen_transform(image=img_H)['image']
            img_L = img_H.copy()
        else:
            img_L = self.get_image_from_path(L_path)

        return img_H, img_L

    def get_patch_from_img(self, img_H, img_L):
        # --------------------------------
        # randomly crop the patch
        # --------------------------------

        if self.n_channels == 3:
            H, W, _ = img_H.shape
        else:
            H, W = img_H.shape

        rnd_h = random.randint(0, max(0, H - self.image_size))
        rnd_w = random.randint(0, max(0, W - self.image_size))
        patch_H = img_H[rnd_h:rnd_h + self.image_size, rnd_w:rnd_w + self.image_size]
        patch_L = img_L[rnd_h:rnd_h + self.image_size, rnd_w:rnd_w + self.image_size]

        return patch_H, patch_L

    def img_fliiper(self, img, random_val, random_vertical):
        # --------------------------------
        # augmentation - flip, rotate
        # --------------------------------

        # randomly horizontal flip the image with probability of 0.5
        if (random_val > 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # randomly vertically flip the image with probability 0.5
        if (random_vertical > 0.5):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img

    def build_transform(self, image_size):
        t = []
        t.append(transforms.ToTensor())  # convert (B, H, W, C) from [0,255] to (B, C, H, W) [0. ,1.]
        return transforms.Compose(t)

    def rescale_tv(self, img, size, type):

        if type == "BILINEAR":
            type = Image.BILINEAR
        elif type == "NEAREST":
            type = Image.NEAREST
        elif type == "BICUBIC":
            type = Image.BICUBIC
        elif type == "LANCZOS":
            type = Image.LANCZOS
        elif type == "HAMMING":
            type = Image.HAMMING

        return transforms.Resize(size, interpolation=type)(img)

    def get_downsample_type(size=1):
        return np.random.choice(
            ["NEAREST", "LANCZOS", "HAMMING", "BILINEAR", "BICUBIC"], size=size, replace=False)
