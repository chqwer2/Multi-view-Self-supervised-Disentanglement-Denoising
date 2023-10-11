import numpy as np 
from PIL import Image
import torch
import shutil
import os
import matplotlib.pyplot as plt
import cv2


### a function that convert network outputs into 8 bit image data
def output_to_image(target_data, output_data, input_data, 
                    plot_img = True, output_8_bit = True, save_img = False,
                    save_dir = 'Results/Snow100K-L/', save_index = 0):
    
    output_data = output_data[0] #discard the batch dimension
    target_data = target_data[0]
    input_data = input_data[0]

    if output_8_bit:
        output_data = output_data * 255
        #print(f"Maximum of output: {np.max(output_data)}")
        output_data = np.clip(output_data, 0, 255).astype(np.uint8)
        target_data = target_data * 255
        #print(f"Maximum of output: {np.max(output_data)}")
        target_data = np.clip(target_data, 0, 255).astype(np.uint8)
        input_data = input_data * 255
        input_data = np.clip(input_data, 0, 255).astype(np.uint8)

    #print(output_data.shape)
    if plot_img:
        output_img = Image.fromarray(output_data)
        target_img = Image.fromarray(target_data)
        input_img = Image.fromarray(input_data)
        if save_img:
            output_img.save(save_dir + str(save_index) + 'output.jpg')
            target_img.save(save_dir + str(save_index) + 'target.jpg')
            input_img.save(save_dir + str(save_index) + 'input.jpg')


    return target_data, output_data

### a function that convert network outputs into 8 bit image data
def save_one_image(img_data, save_dir = "", img_name = "", save_index = 0):
    
    img = np.clip(img_data[0] * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)
            
    img.save(os.path.join(save_dir, img_name))
 


### calculate PSNR between a pair of images
def myPSNR(tar_img, prd_img):
    imdff = np.clip(prd_img,0,1) - np.clip(tar_img,0,1)
    mse = np.mean((imdff**2))
    ps = 20*np.log10(1/np.sqrt(mse)) #MAXf is 1 since our range is from 0 to 1
    return ps

### calculate PSNR between a pair of images
#def myPSNR(tar_img, prd_img):
    #imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    #rmse = (imdff**2).mean().sqrt()
    #ps = 20*torch.log10(1/rmse) #MAXf is 1 since our range is from 0 to 1
    #return ps

### calcuate PSNR for each image in a batch
def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)


def mySSIM(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(mySSIM(img1[..., i], img2[..., i]))

    return np.array(ssims).mean()


def batch_SSIM(img1, img2):
    SSIM = []
    for im1, im2 in zip(img1, img2):
        # preprocessing: scale to 255
        im1 = (im1 * 255.0).round().astype(np.uint8) 
        im2 = (im2 * 255.0).round().astype(np.uint8)

        ssim = calculate_ssim(im1, im2)
        SSIM.append(ssim)
    return sum(SSIM)/len(SSIM)


### save model during training
def save_ckp(state, is_best, checkpoint_dir):
    if is_best:
        f_path = checkpoint_dir 
        torch.save(state, f_path)
    else:
        f_path = checkpoint_dir
        torch.save(state, f_path)


### load checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if('best_error' in checkpoint):
        best_error = checkpoint['best_error']
    else:
        best_error = None
    return model, optimizer, checkpoint['epoch'],\
        checkpoint['error_list'], best_error


### return the base directory
def base_path():
    return os.path.dirname(os.path.abspath(__file__))
    

#graph network error
def graph_error(error_list, name):
    if name[-4:] != ".png":
        if name != "":
            raise Exception("Suffix of file type is needed")
    save_dir = "Losses/" + name
    x = np.arange(len(error_list))
    y = np.asarray(error_list)
    plt.plot(x, y)
    plt.ylabel("Error")
    plt.xlabel("Epoches")
    if name != "":
        plt.savefig(save_dir)
    plt.show()
    
    
    
    