import random
from contextlib import suppress

import numpy as np
import torch
import math

def psnr(ori_img, con_img):
    print()
    print(torch.max(ori_img))
    print(torch.max(con_img))
    # 해당 이미지의 최대값 (채널 최대값 - 최솟값)
    max_pixel = 1.0

    # MSE 계산
    mse = torch.mean((ori_img - con_img) ** 2)  # torch.mean으로 변경
    print(mse)

    if mse == 0:
        return 100
    
    # PSNR 계산
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))  # torch.log10과 torch.sqrt로 변경
    
    return psnr


def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def get_parameter_nums(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
