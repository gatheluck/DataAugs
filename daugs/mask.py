import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import random

import math
import torch


def sample_hard_square_mask(im_size: int, window_size: int):
    """
    Args:
    - im_size (int): size of image.
    - window_size (int): size of window. if -1, return full size mask.

    Return:
    - mask (torch.BoolTensor): hard squre mask.
    """
    assert im_size >= 2
    assert (1 <= window_size <= im_size) or (window_size == -1)

    # if window_size == -1, return all True mask.
    if window_size == -1:
        return torch.ones(im_size, im_size, dtype=torch.bool)

    mask = torch.zeros(im_size, im_size, dtype=torch.bool)  # all elements are False

    # sample window center. if window size is odd, sample from pixel position. if even, sample from grid position.
    window_center_h = random.randrange(0, im_size) if window_size % 2 == 1 else random.randrange(0, im_size + 1)
    window_center_w = random.randrange(0, im_size) if window_size % 2 == 1 else random.randrange(0, im_size + 1)

    for idx_h in range(window_size):
        for idx_w in range(window_size):
            h = window_center_h - math.floor(window_size / 2) + idx_h
            w = window_center_w - math.floor(window_size / 2) + idx_w

            if (0 <= h < im_size) and (0 <= w < im_size):
                mask[h, w] = True

    return mask


def sample_gaussian_circle_mask(im_size: int, sigma: float):
    """
    Args:
    - im_size (int): size of image.
    - sigma (float): variance of Gaussian.

    Return:
    - mask (torch.FloatTensor): soft circle mask.
    """

    def _calc_G(x: int, mu: float, sigma: float):
        expornet = torch.FloatTensor([-1.0 * ((x - mu) ** 2 / (2.0 * (sigma ** 2)))])
        return torch.exp(expornet)

    assert im_size >= 2
    assert sigma > 0

    mask = torch.zeros(im_size, im_size, dtype=torch.float)  # all elements are False

    # sample center of Gaussian.
    mu_h = random.randrange(0, im_size)
    mu_w = random.randrange(0, im_size)

    for idx_h in range(im_size):
        for idx_w in range(im_size):
            Gh = _calc_G(idx_h, mu_h, sigma)
            Gw = _calc_G(idx_w, mu_w, sigma)
            mask[idx_h, idx_w] = Gh * Gw  # elementwise multiplication

    return mask


if __name__ == '__main__':
    import torchvision
    print(sample_hard_square_mask(6, 2))
    print(sample_hard_square_mask(6, 3))
    print(sample_hard_square_mask(5, 2))
    print(sample_hard_square_mask(5, 3))

    print(sample_gaussian_circle_mask(32, 8))
    torchvision.utils.save_image(sample_gaussian_circle_mask(32, 8), '../logs/gaussian_mask_32.png')
    print(sample_gaussian_circle_mask(224, 32))
    torchvision.utils.save_image(sample_gaussian_circle_mask(224, 32), '../logs/gaussian_mask_224.png')