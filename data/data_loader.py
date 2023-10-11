import numpy as np
import os
from data.dataset_base import BaseDataset
import torch
from PIL import Image
from torchvision import transforms

import os
import math
from lib import utils_image as util
from data.mixup import image_mixup
import random

from options import utils_option as option
import lib.utils_noise as rn
from data.noise2void.masker import generate_mask as n2v_generate_mask
from lib.utils_transform import get_transforms




class MedDataset(BaseDataset):
    def __init__(self, opt, dataset_type="train"):
        super(MedDataset, self).__init__()

        self.opt = opt
        self.datasets_opt = opt["datasets"]
        self.dataset_type = dataset_type
        self.DEBUG = False

        self.combine_datasets_path()
        self.get_parms()
        self.albumen_transform = get_transforms(self.dataset_type, self.resize_size)
        self.img_transforms = self.build_transform(self.image_size)


    def __len__(self):
        return len(self.paths_H)

    def apply_mixup(self, patch_H, patch_L, index):
        # --------------------------------
        # randomly Mixup
        # --------------------------------

        if self.opt["datasets"]["use_mixup"]:
            if np.random.random() < self.opt["datasets"]["use_mixup"]:
                mix_index = np.random.randint(low=0, high=self.__len__() - 1)

                mix_img_H, mix_img_L = self.get_img_by_index(mix_index)
                mix_patch_H, mix_patch_L = self.get_patch_from_img(mix_img_H, mix_img_L)

                patch_H = image_mixup(patch_H, mix_patch_H)
                patch_L = image_mixup(patch_L, mix_patch_L)

        assert patch_L.shape == patch_H.shape, f"[{index}]: do not have the same dimension"
        return patch_H, patch_L

    def apply_flip(self, patch_H, patch_L):
        # --------------------------------
        # randomly Flip
        # --------------------------------
        patch_L = Image.fromarray(patch_L)
        patch_H = Image.fromarray(patch_H)

        random_val = np.random.uniform(0, 1)
        random_vertical = np.random.uniform(0, 1)

        img_L1 = self.img_fliiper(patch_L, random_val, random_vertical)
        img_H = self.img_fliiper(patch_H, random_val, random_vertical)
        return img_H, img_L1


    def r2r_noise(self, img_L1, img_L2, img_H, sigma):
        img_train = rn.gaussian(img_L1, sigma / 255.0)

        eps = sigma / 255.
        alpha = 0.5
        pert = eps * torch.FloatTensor(img_H.size()).normal_(mean=0, std=1.)  # .cuda()
        img_L1 = img_train + alpha * pert
        img_H = img_train - pert / alpha
        img_L2 = img_L1
        return img_L1, img_L2, img_H


    def gaussian_noise(self, img_L1, img_L2, sigma):
        img_L1 = rn.gaussian(img_L1, sigma / 255.0)
        img_L2 = rn.gaussian(img_L2, sigma / 255.0)
        return img_L1, img_L2


    def noisepool(self, img_L1, img_L2, sigma, p = 0.5):

        if random.random() < p:
            img_L1 = rn.gaussian(img_L1, sigma / 255.0)
        else:
            img_L1 = rn.select_noise(img_L1, self.noise_type, sigma / 255.0, self.noise_num
                                 , sp_noise=self.opt["corruption"]["sp_amount"])

        if random.random() < p:
            img_L2 = rn.gaussian(img_L2, sigma / 255.0)
        else:
            img_L2 = rn.select_noise(img_L2, self.noise_type, sigma / 255.0, self.noise_num
                                 , sp_noise=self.opt["corruption"]["sp_amount"])

        return img_L1, img_L2

    def apply_noise(self, img_L1, img_L2, img_H):
        # --------------------------------
        # randomly Add Noise
        # --------------------------------
        if isinstance(self.sigma, list):
            sigma = np.random.randint(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma

        # transform each data to tensor with value range 0-1
        img_L1 = self.img_transforms(img_L1).float()  # ToTensor
        img_L2 = self.img_transforms(img_L2).float()
        img_H = self.img_transforms(img_H).float()


        # Standardize the shape of the image
        if len(img_H.shape) < 3:
            img_H = np.expand_dims(img_H, 0)
            img_L1 = np.expand_dims(img_L1, 0)
            img_L2 = np.expand_dims(img_L2, 0)


        # if add noise
        if np.random.random() < self.noise_p:

            if self.opt["train_type"] == "r2r":
                img_L1, img_L2, img_H = self.r2r_noise(img_L1, img_L2, img_H, sigma)

            elif self.opt["noise_pool"]:
                if np.random.random() < 0.5:
                    img_L1, img_L2 = self.noisepool(img_L1, img_L2, sigma)
                else:
                    img_L1, img_L2 = self.gaussian_noise(img_L1, img_L2, sigma)

            else:
                # Gaussian noise only
                img_L1, img_L2 = self.gaussian_noise(img_L1, img_L2, sigma)

        return img_L1, img_L2, img_H


    def check_4channels(self, img_L1, img_L2):
        if len(img_L1.shape) < 4:
            img_L1 = img_L1.unsqueeze(0)
            img_L2 = img_L2.unsqueeze(0)

        return img_L1, img_L2

    def __getitem__(self, index):
        output_dict = {}

        img_H, img_L = self.get_img_by_index(index)

        
        if self.dataset_type == "train":

            patch_H, patch_L = self.get_patch_from_img(img_H, img_L)

            mode = random.randint(0, 7)
            patch_H, patch_L = self.apply_mixup(patch_H, patch_L, index)

            img_H, img_L1 = self.apply_flip(patch_H, patch_L)
            img_L2 = img_L1.copy()


            img_L1, img_L2, img_H = self.apply_noise(img_L1, img_L2, img_H)

            img_L1, img_L2 = self.check_4channels(img_L1, img_L2)

            # print("after check_4channels:", img_L1.shape, img_L2.shape, img_H.shape)

            # print("img_L1:")
        elif self.dataset_type == "test":
            # make PyTorch happy, it accepts PIL image only
            img_L = Image.fromarray(img_L)
            img_H = Image.fromarray(img_H)

            if self.n_channels < 3:
                img_H = img_H.convert("L")
                img_L = img_L.convert("L")


            # transform each data to torch tensors
            img_L1 = self.img_transforms(img_L)   # ToTensor
            img_H = self.img_transforms(img_H)

            if len(img_H.shape) < 3:
                img_H = np.expand_dims(img_H, 0)
                img_L1 = np.expand_dims(img_L1, 0)

            # --------------------------------
            # randomly Impainting
            # --------------------------------
            if np.random.random() < self.impainting_p:
                impainting_mask = np.random.uniform(size=[1, img_H.shape[1], img_H.shape[2]] )
                mask_pixel = torch.from_numpy(impainting_mask > self.impainting_rate)  # Main
                img_L1 = mask_pixel * img_L1 * 0.7
                output_dict["mask_pixel_1"] = mask_pixel



            if np.random.random() < self.noise_p:
                # img_L1 += np.random.normal(0, self.sigma_test / 255.0, img_L1.shape)
                img_L1 = rn.gaussian(img_L1, self.sigma_test / 255.0)

            img_L2 = img_L1


        output_dict["L1"], output_dict["L2"] = img_L1.float(), img_L2.float()
        output_dict["H"] = img_H.float()
        return output_dict





class MedTestDataset(BaseDataset):
    def __init__(self, opt, datasets=["CBSD68"], noise_types=["gaussian@25"]):
        super(MedTestDataset, self).__init__()

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

        self.noise_p = self.opt["corruption"]["noise"]
        self.rain_p = self.opt["corruption"]["rain"]
        self.blur_p = self.opt["corruption"]["blur"]

        self.impainting_p = self.opt["corruption"]["impainting"]
        self.impainting_rate = self.opt["corruption"]["impainting_rate"]
        self.superresolution_p = self.opt["corruption"]["superresolution"]
        self.superresolution_scale = self.opt["corruption"]["superresolution_scale"]

        self.img_transforms = self.build_transform(self.image_size)



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


        img_H = img_H[:, :, :3 ]    #
        img_L = img_L[:, :, :3]

        print("img_H:", img_H.shape)

        # make PyTorch happy, it accepts PIL image only
        img_L = Image.fromarray(img_L)
        img_H = Image.fromarray(img_H)

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
            # print("")
            img_L1 = np.clip( img_L1, 0, 1)
            
        img_L2 = img_L1
    
        return {'L1': img_L1.float(), 'L2': img_L2.float(), 'H': img_H.float()}


class MedNoisyOnlyDataset(BaseDataset):
    def __init__(self, opt, datasets=["CBSD68"]):
        super(MedNoisyOnlyDataset, self).__init__()

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
