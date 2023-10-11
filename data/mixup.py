import numpy as np



def image_mixup(img1, img2):
    alpha = 0.5
    lam = np.random.beta(alpha, alpha)
    
    return np.array(lam * img1 + (1 - lam) * img2, dtype = np.uint8)

