import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

def l1(pred, target):
    return F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)

def l2(pred, target):
    return F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)

def l1_ssim(pred, target, alpha=0.7):
    return alpha * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + (1- alpha) * (1 - ssim(pred, target, data_range=1, size_average=False))

def l2_ssim(pred, target, alpha=0.7):
    return alpha * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + (1- alpha) * (1 - ssim(pred, target, data_range=1, size_average=False))