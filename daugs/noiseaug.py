import os
import sys
import random

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch

from daugs.mask import sample_hard_square_mask
from daugs.mask import sample_gaussian_circle_mask

from submodules.FourierHeatmap.fhmap.fourier_heatmap import AddFourierNoise


class NoiseBasedAugmentation():
    def __init__(self):
        pass

    def __call__(self, x):
        pass

    def add_noise(self, x: torch.tensor, noise: torch.tensor, mask: torch.tensor):
        assert x.shape == noise.shape == mask.shape

        noised_x = torch.clamp(x + noise, 0.0, 1.0)

        x_a = mask * noised_x
        x_b = (torch.ones_like(mask) - mask) * x
        x_mix = torch.clamp(x_a + x_b, min=0.0, max=1.0)

        return x_mix


class AddPatchGaussian(NoiseBasedAugmentation):
    """
    implementation of PatchGaussian (arXiv 2019): https://arxiv.org/abs/1906.02611
    """
    def __init__(self,
                 patch_size: int,
                 max_scale: float,
                 randomize_patch_size: bool,
                 randomize_scale: bool,
                 **kwargs):
        """
        Args:
        - patch_size: size of patch. if -1, it means all image
        - max_scale: max scale size. this value should be in [1, 0]
        - randomize_patch_size: whether randomize patch size or not
        - randomize_scale: whether randomize scale or not
        """
        assert (patch_size >= 1) or (patch_size == -1)
        assert 0.0 <= max_scale <= 1.0

        self.patch_size = patch_size
        self.max_scale = max_scale
        self.randomize_patch_size = randomize_patch_size
        self.randomize_scale = randomize_scale

    def __call__(self, x: torch.tensor):
        c, w, h = x.shape[-3:]

        assert c == 3
        assert h >= 1 and w >= 1
        assert h == w

        # randomize scale and patch_size
        scale = random.uniform(0, 1) * self.max_scale if self.randomize_scale else self.max_scale
        patch_size = random.randrange(1, self.patch_size + 1) if self.randomize_patch_size else self.patch_size
        mask = sample_hard_square_mask(h, patch_size).repeat(c, 1, 1)

        gaussian = torch.normal(mean=0.0, std=scale, size=(c, w, h))
        x_mix = self.add_noise(x, gaussian, mask)

        return x_mix


class AddSmoothGaussian(NoiseBasedAugmentation):
    """
    combine the concept of PatchGaussian and SmoothMix.
    - PatchGaussian (arXiv 2019): https://arxiv.org/abs/1906.02611
    - SmoothMix (CVPR Workshop 2020): https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf
    """
    def __init__(self,
                 max_scale: float,
                 randomize_scale: bool,
                 sigmas: list,
                 **kwargs):
        """
        Args:
        - patch_size: size of patch. if -1, it means all image
        - max_scale: max scale size. this value should be in [1, 0]
        - randomize_patch_size: whether randomize patch size or not
        - randomize_scale: whether randomize scale or not
        """
        assert 0.0 <= max_scale <= 1.0
        assert len(sigmas) > 0

        self.max_scale = max_scale
        self.randomize_scale = randomize_scale
        self.sigmas = sigmas

    def __call__(self, x: torch.tensor):
        c, w, h = x.shape[-3:]

        assert c == 3
        assert h >= 1 and w >= 1
        assert h == w

        # sigma = random.choices(self.sigmas, k=1)[0]
        sigma = random.uniform(min(self.sigmas), max(self.sigmas))
        mask = sample_gaussian_circle_mask(h, sigma)[None, :, :].repeat(3, 1, 1)

        scale = random.uniform(0, 1) * self.max_scale if self.randomize_scale else self.max_scale
        gaussian = torch.normal(mean=0.0, std=scale, size=(c, w, h))

        x_mix = self.add_noise(x, gaussian, mask)
        return x_mix


class AddPatchFourier(NoiseBasedAugmentation):
    def __init__(self,
                 patch_size: int,
                 randomize_patch_size: bool,
                 eps: float,
                 **kwargs):
        """
        Args:
        - patch_size: size of patch. if -1, it means all image
        - max_scale: max scale size. this value should be in [1, 0]
        - randomize_patch_size: whether randomize patch size or not
        - randomize_scale: whether randomize scale or not
        """
        assert (patch_size >= 1) or (patch_size == -1)
        assert eps >= 0.0

        self.patch_size = patch_size
        self.randomize_patch_size = randomize_patch_size
        self.eps = eps

    def __call__(self, x: torch.tensor):
        c, w, h = x.shape[-3:]

        assert c == 3
        assert h >= 1 and w >= 1
        assert h == w

        patch_size = random.randrange(1, self.patch_size + 1) if self.randomize_patch_size else self.patch_size
        mask = sample_hard_square_mask(h, patch_size).repeat(c, 1, 1)

        max_h_index = int((h - 1) // 2)
        max_w_index = int((w - 1) // 2)
        h_index = random.randrange(-max_h_index, max_h_index)
        w_index = random.randrange(-max_w_index, max_w_index)

        fourier_noise = AddFourierNoise.calc_fourier_noise(c, h, w, h_index, w_index, self.eps)
        x_mix = self.add_noise(x, fourier_noise, mask)

        return x_mix


class AddSmoothFourier(NoiseBasedAugmentation):
    def __init__(self,
                 sigmas: list,
                 eps: float,
                 **kwargs):
        """
        Args:
        - patch_size: size of patch. if -1, it means all image
        - max_scale: max scale size. this value should be in [1, 0]
        - randomize_patch_size: whether randomize patch size or not
        - randomize_scale: whether randomize scale or not
        """
        assert len(sigmas) > 0
        assert eps >= 0.0

        self.sigmas = sigmas
        self.eps = eps

    def __call__(self, x: torch.tensor):
        c, w, h = x.shape[-3:]

        assert c == 3
        assert h >= 1 and w >= 1
        assert h == w

        # sigma = random.choices(self.sigmas, k=1)[0]
        sigma = random.uniform(min(self.sigmas), max(self.sigmas))
        mask = sample_gaussian_circle_mask(h, sigma)[None, :, :].repeat(3, 1, 1)

        max_h_index = int((h - 1) // 2)
        max_w_index = int((w - 1) // 2)
        h_index = random.randrange(-max_h_index, max_h_index)
        w_index = random.randrange(-max_w_index, max_w_index)

        fourier_noise = AddFourierNoise.calc_fourier_noise(c, h, w, h_index, w_index, self.eps)
        x_mix = self.add_noise(x, fourier_noise, mask)

        return x_mix


if __name__ == '__main__':
    import torchvision
    augmentors = {
        'patch_gaussian': AddPatchGaussian(patch_size=16, max_scale=0.5, randomize_patch_size=True, randomize_scale=True),
        'smooth_gaussian': AddSmoothGaussian(max_scale=0.5, randomize_scale=True, sigmas=[0.25, 0.5]),
        'patch_fourier': AddPatchFourier(patch_size=16, randomize_patch_size=True, eps=16.0),
        'smooth_fourier': AddSmoothFourier(sigmas=[0.25], eps=16.0)
    }

    for augmentor_name, augmentor in augmentors.items():
        transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
            augmentor
        ])
        dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        for (x, t) in loader:
            x, t = x.cuda(), t.cuda()
            torchvision.utils.save_image(x, '../logs/{augmentor}.png'.format(augmentor=augmentor_name))
            break
