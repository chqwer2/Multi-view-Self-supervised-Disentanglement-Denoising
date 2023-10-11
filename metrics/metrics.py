from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
import lpips

import ignite.distributed as idist
from ignite.metrics import FID, InceptionScore, SSIM, PSNR

# pip install lpips
# pip install pytorch-ignite

def eval_step(engine, batch):
    return batch

from torch import nn
# from metrics.InceptionV3 import *


class metrics(nn.Module):
    def __init__(self):
        super().__init__()
        # closer to "traditional" perceptual loss, when used for optimization
        self.compute_lpips = lpips.LPIPS(net='vgg')
        self.evaluator = Engine(eval_step)

        device = idist.device()

        #  num_features=dims, feature_extractor=wrapper_model
        # , output_transform=lambda.py x: x[0]
        self.compute_fid = FID(device=device)
        self.compute_is = InceptionScore(device=device)
        self.compute_ssim = SSIM(device=device, data_range=1.0)
        self.compute_psnr = PSNR(device=device, data_range=1.0)

        self.compute_psnr.attach(self.evaluator, 'psnr')
        # self.compute_fid.attach(self.evaluator, 'fid')
        self.compute_ssim.attach(self.evaluator, 'ssim')
        # self.compute_is.attach(self.evaluator, 'is')

        # loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores

    def forward(self, gt, pred):
        # 0 ~ 1
        lpips_score = self.compute_lpips(gt, pred)
        state = self.evaluator.run([[gt, pred]])
        return {#"fid": state.metrics["fid"], "lpips": lpips_score, "is": state.metrics["is"],
                "ssim": state.metrics["ssim"], "psnr": state.metrics["psnr"]}