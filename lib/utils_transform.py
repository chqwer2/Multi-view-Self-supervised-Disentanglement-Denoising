from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, SmallestMaxSize,
    CenterCrop, Resize, RandomCrop, GaussianBlur, JpegCompression, Downscale, ElasticTransform
)

'''
# --------------------------------------------
# Calvin Hao Chen (chqwer2.github.io)
# 19/Jan/2023
# --------------------------------------------
'''



def get_transforms(type, img_size):

    if type == 'train':
        return Compose([
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.25),
            RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),

            OneOf([
                GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.0),

            CoarseDropout(max_holes=8, max_height=img_size // 20, max_width=img_size // 20,
                          min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),

        ], p=1.0, additional_targets={'image2': 'image'})


    elif type == 'light_train':
        return Compose([
            ShiftScaleRotate(shift_limit=0.025, scale_limit=0.02, rotate_limit=5, p=0.5),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.25),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ], p=1.0, additional_targets={'image2': 'image'})

    else:
        return Compose([], additional_targets={'image2': 'image'})


