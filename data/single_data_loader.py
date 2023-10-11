import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms

import os
import math
from utils import utils_image as util
import torch.nn.functional as F
from data.motion_blur.blur_image import BlurImage
from data.addrain.addrain import addrain
from data.SynaRain.SynaRainer import SynaRainer
from data.mixup import image_mixup
import random

from options import utils_option as option
import data.random_noise as rn
from data.noise2void.masker import generate_mask as n2v_generate_mask

class MediaDataset(Dataset):
    def __init__(self, opt, dataset_type="train"):
        super(MediaDataset, self).__init__()

        self.opt = opt
        datasets_opt = opt["datasets"]

        self.get_dataset = datasets_opt[dataset_type + "_dataset"]
        self.dataset_file = option.json_parse(self.opt["dataset_file"])[self.get_dataset]

        self.dataset_type = dataset_type
        self.shift = datasets_opt["shift"]

        self.n_channels = datasets_opt['n_channels'] if datasets_opt['n_channels'] else 3
        self.image_size = datasets_opt['H_size'] if datasets_opt['H_size'] else 64
        self.window_size = self.opt["netG"]["window_size"]

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        # self.paths_H_filename = self.dataset_file['dataroot_H_filename']
        # self.paths_L_filename = self.dataset_file['dataroot_L_filename']
        #
        # self.paths_H = util.get_image_paths(os.path.join(self.dataset_file['dataroot_H'], self.paths_H_filename[0]))
        # if self.dataset_file['dataroot_L'] != None:
        #     self.paths_L = util.get_image_paths(os.path.join(self.dataset_file['dataroot_L'], self.paths_L_filename[0]))
        # else:
        #     self.paths_L = self.paths_H

        # ------------------------------------
        # Probability
        # ------------------------------------

        self.impainting_p = self.opt["corruption"]["impainting"]
        self.impainting_rate = self.opt["corruption"]["impainting_rate"]
        self.superresolution_p = self.opt["corruption"]["superresolution"]
        self.superresolution_scale = self.opt["corruption"]["superresolution_scale"]

        self.file_name = self.opt["datasets"]["single_image"]


        self.img_transforms = self.build_transform(self.image_size)

        self.img_H = self.get_img_by_name()
        impainting_mask = np.random.uniform(
              size=self.img_H.shape)
        mask_pixel = (impainting_mask > self.impainting_rate)
        self.mask_pixel = mask_pixel
        self.raw_img_H = self.img_H.copy()

        self.img_H = (self.img_H * self.mask_pixel)   #.detach().numpy()

    # image transform
    def build_transform(self, image_size):
        t = []
        t.append(transforms.ToTensor())  # convert (B, H, W, C) from [0,255] to (B, C, H, W) [0. ,1.]
        return transforms.Compose(t)

    def __len__(self):
        return 1

    def get_img_by_name(self):
        H_path = os.path.join(self.dataset_file['dataroot_H'], self.file_name)
        img_H = np.asarray(Image.open(H_path))

        return img_H
    
    def get_patch_from_img(self, img_H, img_L):
        # --------------------------------
        # randomly crop the patch
        # --------------------------------
            
        H, W, _ = img_H.shape
        rnd_h = random.randint(0, max(0, H - self.image_size))
        rnd_w = random.randint(0, max(0, W - self.image_size))
        patch_H = img_H[rnd_h:rnd_h + self.image_size, rnd_w:rnd_w + self.image_size, :]
        patch_L = img_L[rnd_h:rnd_h + self.image_size, rnd_w:rnd_w + self.image_size, :]
        
            
        return patch_H, patch_L
    
    def img_fliiper(self, img, random_val, random_vertical):
        # --------------------------------
        # augmentation - flip, rotate
        # --------------------------------
        # randomly horizontal flip the image with probability of 0.5
        if (random_val > 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # randomly vertically flip the image with probability 0.5
        if random_vertical > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
        return img
                        
    def __getitem__(self, index):

        output_dict = {}

        if self.dataset_type == "train":

            # patch_L = Image.fromarray(patch_L)
            patch_H = Image.fromarray(self.img_H)  # with hole
            mask_pixel = Image.fromarray(self.mask_pixel)

            # make PyTorch happy, it accepts PIL image only
            random_val = np.random.uniform(0, 1)
            random_vertical = np.random.uniform(0, 1)

            img_H = self.img_fliiper(patch_H, random_val, random_vertical)
            mask_pixel = self.img_fliiper(mask_pixel, random_val, random_vertical)

            img_H = self.img_transforms(img_H).float()

            mask_pixel = torch.from_numpy(np.asarray(mask_pixel)).float()

            # -------------------
            # Impainting
            # -------------------
            if np.random.random() < self.impainting_p:

                impainting_mask = np.random.uniform(size=[1, img_H.shape[1], img_H.shape[2]])
                # impainting_mask = np.repeat(impainting_mask, [1, 1, 3])

                if self.opt["train_type"] == "n2c":
                    mask_pixel = (impainting_mask > self.impainting_rate) * 0.7
                    mask_pixel = torch.from_numpy(mask_pixel)
                    img_L1 = img_H * mask_pixel
                    output_dict["mask_pixel_1"] = 1 - mask_pixel  # 1-0.7 / 0.3 or 1.0
                    img_L2 = img_L1

            if len(img_L1.shape) < 4:
                        img_L1 = img_L1.unsqueeze(0)
                        img_L2 = img_L2.unsqueeze(0)
            # print("img_L1 shape:", img_L1.shape)

        elif self.dataset_type == "test":
            # make PyTorch happy, it accepts PIL image only
            img_H = Image.fromarray(self.raw_img_H.copy())

            # transform each data to torch tensors
            img_H = self.img_transforms(img_H)


            # --------------------------------
            # randomly Impainting
            # --------------------------------
            if np.random.random() < self.impainting_p:
                impainting_mask = np.random.uniform(size=[1, img_H.shape[1], img_H.shape[2]])

                mask_pixel = (impainting_mask > self.impainting_rate) * 0.7

                mask_pixel = torch.from_numpy(np.asarray(mask_pixel)).float()

                img_L1 = mask_pixel * img_H.clone()
                output_dict["mask_pixel_1"] = mask_pixel


            img_L2 = img_L1
        # , 'H_path': H_path, 'L_path': L_path
        output_dict["L1"], output_dict["L2"] = img_L1.float(), img_L2.float()
        output_dict["H"] = img_H.float()
        return output_dict





class MediaTestDataset(Dataset):
    def __init__(self, opt, datasets=["CBSD68"], noise_types=["gaussian@25"]):
        super(MediaTestDataset, self).__init__()

        self.opt = opt
        datasets_opt = opt["datasets"]

        self.dataset_file = option.json_parse(self.opt["dataset_file"])[datasets]

        # Data Corruption Type Handle
        self.noise_type = []
        self.noise_sigma = []
        
        for noise in noise_types:
            
            self.noise_type.append(noise.split('@')[0])
            try: 
                self.noise_sigma.append(int(noise.split('@')[1]))
            except:
                self.noise_sigma.append(0)
            # print("self.noise_type: ", self.noise_type)
        

        self.n_channels = datasets_opt['n_channels'] if datasets_opt['n_channels'] else 3
        self.image_size = datasets_opt['H_size'] if datasets_opt['H_size'] else 64
        self.window_size = self.opt["netG"]["window_size"]

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H_filename = self.dataset_file['dataroot_H_filename']
        self.paths_L_filename = self.dataset_file['dataroot_L_filename']

        self.paths_H = util.get_image_paths(os.path.join(self.dataset_file['dataroot_H'], self.paths_H_filename[0]))
        if self.dataset_file['dataroot_L'] != None:
            self.paths_L = util.get_image_paths(os.path.join(self.dataset_file['dataroot_L'], self.paths_L_filename[0]))
        else:
            self.paths_L = None

        # ------------------------------------
        # Probability
        # ------------------------------------
        self.BlurImage = BlurImage()


        self.noise_p = self.opt["corruption"]["noise"]
        self.rain_p = self.opt["corruption"]["rain"]
        self.blur_p = self.opt["corruption"]["blur"]

        self.impainting_p = self.opt["corruption"]["impainting"]
        self.impainting_rate = self.opt["corruption"]["impainting_rate"]
        self.superresolution_p = self.opt["corruption"]["superresolution"]
        self.superresolution_scale = self.opt["corruption"]["superresolution_scale"]

        self.img_transforms = self.build_transform(self.image_size)
        try:
            self.SynaRainer = SynaRainer(img_size=datasets_opt['H_size'])
        except:
            print("SynaRainer cannot be loaded in")

        if self.rain_p > 0:
            self.RainImage = addrain()

    # image transform
    def build_transform(self, image_size):
        t = []
        t.append(transforms.ToTensor())  # convert (B, H, W, C) from [0,255] to (B, C, H, W) [0. ,1.]
        return transforms.Compose(t)

    def __len__(self):
        return len(self.paths_H)

    def get_img_by_index(self, index):
        H_path = self.paths_H[index]
        if self.paths_L:
            # print("Here...")
            L_path = self.paths_L[index]
        else:
            L_path = self.paths_H[index].replace("/" + self.paths_H_filename[1], "/" + self.paths_L_filename[1])

        img_H = np.asarray(Image.open(H_path))
        img_L = np.asarray(Image.open(L_path))
        return img_H, img_L
    
    

    def __getitem__(self, index):

        img_H, img_L = self.get_img_by_index(index)
      
        # make PyTorch happy, it accepts PIL image only
        img_L = Image.fromarray(img_L)
        img_H = Image.fromarray(img_H)

        # if np.random.random() < self.blur_p:
        #     img_L = self.BlurImage.blur_image(img_L)



        # transform each data to torch tensors
        img_L1 = self.img_transforms(img_L)   # ToTensor
        img_H = self.img_transforms(img_H)

        if self.opt["datasets"]["real_noise"]:
            print("Using real noise")
            return {'L1': img_L1.float(),
                    'L2': img_L1.float(), 'H': img_H.float()}


        noise_ids = list(range(len(self.noise_type)))
        np.random.shuffle(noise_ids)
        
        for noise_id in noise_ids:
            print("test noise type in data loader:", self.noise_type[noise_id], 
                  " ", self.noise_sigma[noise_id] )
            
            sigma = self.noise_sigma[noise_id] / 255.0
            img_L1 = rn.select_one_noise(img_L1, self.noise_type[noise_id], sigma)
            
        img_L2 = img_L1
    
        return {'L1': img_L1.float(), 'L2': img_L2.float(), 'H': img_H.float()}


class MediaNoisyOnlyDataset(Dataset):
    def __init__(self, opt, datasets=["CBSD68"]):
        super(MediaNoisyOnlyDataset, self).__init__()

        self.opt = opt
        datasets_opt = opt["datasets"]

        self.dataset_file = option.json_parse(self.opt["dataset_file"])[datasets]

        self.n_channels = datasets_opt['n_channels'] if datasets_opt['n_channels'] else 3
        self.image_size = datasets_opt['H_size'] if datasets_opt['H_size'] else 64
        self.window_size = self.opt["netG"]["window_size"]

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_L_filename = self.dataset_file['dataroot_L_filename']
        self.paths_L = util.get_image_paths(os.path.join(self.dataset_file['dataroot_L'], self.paths_L_filename[0]))
        self.img_transforms = self.build_transform(self.image_size)


    # image transform
    def build_transform(self, image_size):
        t = []
        t.append(transforms.ToTensor())  # convert (B, H, W, C) from [0,255] to (B, C, H, W) [0. ,1.]
        return transforms.Compose(t)

    def __len__(self):
        return len(self.paths_L)

    def get_img_by_index(self, index):
        L_path = self.paths_L[index]
        img_L = np.asarray(Image.open(L_path))
        return img_L

    def __getitem__(self, index):

        img_L = self.get_img_by_index(index)
        
        if np.max(img_L) > 256:
                img_L = img_L/ 256     # H x W x C) in the range [0, 255] 
                img_L = np.asarray(img_L, np.uint8) 
                
        img_L = Image.fromarray(img_L)           
        img_L = self.img_transforms(img_L)

        return {'L1': img_L.float(),
                'L2': img_L.float(), 'H': img_L.float()}
