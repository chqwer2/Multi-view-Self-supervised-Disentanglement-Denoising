import torch
import numpy as np

rng = np.random.default_rng(seed=42)

'''
# --------------------------------------------
# Hao Chen (chqwer2.github.io)
# 19/Jan/2023
# --------------------------------------------
'''


def _bernoulli(p, shape):
    return torch.rand(shape) <= p

def gaussian(img, sigma):
    return img + torch.FloatTensor(img.shape).normal_(mean=0, std=sigma)

def gaussian_localvar(img):
    # var ** 0.5
    noise = np.random.normal(0, torch.sqrt(img*255).numpy(), img.shape) / 255
    return img + noise #torch.FloatTensor(img.shape).normal_(mean=0, std=torch.sqrt(img).numpy())

def salt_pepper(img, amount=0.05, salt_vs_pepper=0.5):

    flipped = _bernoulli(amount, img.shape)
    salted = _bernoulli(salt_vs_pepper, img.shape)
    peppered = ~salted

    img[flipped & salted] = 1
    img[flipped & peppered] = 0

    return img

def salt_pepper_3(img, amount=0.05, salt_vs_pepper=0.3):

    flipped = _bernoulli(amount, img.shape[:2]).unsqueeze(-1).repeat(1, 1, 3)
    salted = _bernoulli(salt_vs_pepper, img.shape[:2]).unsqueeze(-1).repeat(1, 1, 3)

    peppered = ~salted

    img[flipped & salted] = 1
    img[flipped & peppered] = 0

    return img, flipped & salted

def salt_pepper_3_torch(img, amount=0.05, salt_vs_pepper=0.3):

    peppered = _bernoulli(amount, img.shape[-2:]).unsqueeze(0).repeat(3, 1, 1)
    img[peppered] = 0

    return img

def poisson(img):
    img_ = (img.clone() * 255).to(torch.uint8)

    vals = torch.unique(img_).shape[0]
    vals = 2 ** torch.ceil(torch.as_tensor(np.log2(vals)))

    img = torch.poisson(img_ * vals) / float(vals) / 255

    return img



def speckle(img, sigma):
    
    return img + img * torch.FloatTensor(img.shape).normal_(mean=0, std=sigma)


def select_one_noise(img, type_, sigma):

    if  type_ == "gaussian":
        img = gaussian(img, sigma)
        
    elif type_ == "poisson":
        img = poisson(img)

    elif type_ == "local_val":
        img = gaussian_localvar(img)

    elif type_ == "s&p":
        img = salt_pepper(img, amount=0.05, salt_vs_pepper=0.5)

    elif type_ == "pepper":

        img = salt_pepper_3_torch(img, amount=sigma*255/100, salt_vs_pepper=0)

    elif type_ == "speckle":
        img = speckle(img, sigma)
    else:
        print("Not this noise type supported:", type_)

    return img


def select_noise(img, type_, sigma, num, sp_noise=[]):
    selected = np.random.choice(type_, num, replace=False)

    for s in selected:
        if   s == "poisson":
            img = poisson(img)

        elif s == "local_val":
            img = gaussian_localvar(img)

        elif s == "s&p":
            amount = sp_noise
            s_amount = np.random.randint(amount[0], amount[1])

            img = salt_pepper(img, amount=s_amount)

        elif s == "speckle":
            img = speckle(img, sigma)

        else:
            print("Not this noise type supported:", s)

    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    import torchvision.transforms as T
    from utils import batch_PSNR, batch_SSIM

    def shrink(img):
        return T.Resize([128, 128])(img.permute(2, 0, 1)).permute(1, 2, 0)

    def re_shrink(img):
        return T.Resize([256, 256])(img.permute(2, 0, 1)).permute(1, 2, 0)

    input_file = "../figure/8.tif.png"


    origin = torch.tensor(plt.imread(input_file))[:,:,:3]
    # Result saved to sample


    print("origin:", origin.shape)  # torch.Size([512, 512, 3])


    # gaussian
    g_25 = gaussian(origin, sigma=25 / 255)
    g_50 = gaussian(origin, sigma=50 / 255)

    # speckle
    s_25 = speckle(origin, sigma=25 / 255)
    s_50 = speckle(origin, sigma=75 / 255)

    # Salt and pepper with ratio 0.1
    sp, sp_noise = salt_pepper_3(origin, amount=0.1)  # amount=0.05,

    # output_data = s_50.numpy()
    # psnr = batch_PSNR(origin.numpy(), output_data, True).item()
    # ssim = batch_SSIM(origin.numpy(), output_data).item()
    # print("psnr, ssim")

    p = poisson(origin)

    lvg = gaussian_localvar(origin)


    sp_noise = torch.clip(sp_noise, 0, 1).numpy() - 0.0001 + 0.0001
    # sp_noise = np.asarray(np.ones_like(sp_noise)[sp_noise], np.uint8)

    # print("sp_noise :", sp_noise )




    plt.imsave("./sample/g_25.jpg", torch.clip(g_25, 0, 1).numpy())
    plt.imsave("./sample/g_50.jpg",  torch.clip(g_50, 0, 1).numpy())
    plt.imsave("./sample/s_25.jpg",  torch.clip(s_25, 0, 1).numpy())
    plt.imsave("./sample/s_50.jpg",  torch.clip(s_50, 0, 1).numpy())

    plt.imsave("./sample/p.jpg",  torch.clip(p, 0, 1).numpy())
    plt.imsave("./sample/sp.jpg",  torch.clip(sp, 0, 1).numpy())
    plt.imsave("./sample/sp_noise.jpg", sp_noise)
    plt.imsave("./sample/lvg.jpg",  torch.clip(lvg, 0, 1).numpy())
    print("Done...")

