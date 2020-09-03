import os
import sys
import random

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import torch

from daugs.mask import sample_hard_square_mask
from daugs.mask import sample_gaussian_circle_mask


class MixAugmentation():
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor, t: torch.Tensor):
        pass

    def _clac_output(self,
                     model: torch.nn,
                     x: torch.tensor,
                     rand_index: torch.tensor,
                     masks: torch.tensor):
        """
        calcutate output of forwarding mixed input.
        mask * x_a + (1.0 - mask) * x_b
        """
        assert x.size(0) == rand_index.size(0) == masks.size(0)
        assert len(x.shape) == len(masks.shape) == 4
        assert len(rand_index.shape) == 1
        assert torch.min(masks) >= torch.min(torch.zeros_like(masks)) and torch.max(masks) <= torch.max(torch.ones_like(masks))

        x_a = masks * x
        x_b = (torch.ones_like(masks) - masks) * x[rand_index]
        x_mix = torch.clamp(x_a + x_b, min=0.0, max=1.0)
        output = model(x_mix)
        return output, x_mix.detach(), x_a.detach(), x_b.detach()

    def _calc_loss(self,
                   output: torch.tensor,
                   t: torch.tensor,
                   rand_index: torch.tensor,
                   lams: torch.tensor,
                   criterion: torch.nn) -> torch.tensor:
        """
        calcurate loss for mix augmentation.
        lamda * (loss for t_a)   + (1.0 - lamda) * (loss for t_b)
        """
        assert output.size(0) == t.size(0) == rand_index.size(0) == lams.size(0)
        assert len(output.shape) == 2
        assert len(t.shape) == len(rand_index.shape) == len(lams.shape) == 1

        t_a, t_b = t, t[rand_index]
        loss = (lams * criterion(output, t_a)) + ((torch.ones_like(lams) - lams) * criterion(output, t_b))
        return loss


class Mixup(MixAugmentation):
    """
    implementation of Mixup (ICLR 2018): https://arxiv.org/abs/1710.09412
    """
    def __init__(self,
                 beta_dist_a: float = 1.0,
                 beta_dist_b: float = 1.0,
                 criterion=torch.nn.CrossEntropyLoss(),
                 device: str = 'cuda'):
        self.beta_dist_a = beta_dist_a
        self.beta_dist_b = beta_dist_b
        self.criterion = criterion
        self.device = device

    def __call__(self, model: torch.nn, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rand_index = torch.randperm(x.size(0))
        lams = self._sample_lambda(x.size(0), self.beta_dist_a, self.beta_dist_b).to(self.device)  # shape of lams is [b]
        masks = lams[:, None, None, None].repeat(1, 3, x.size(-2), x.size(-1))  # shape of masks are [b, 3, h, w]

        output, x_mix, x_a, x_b = self._clac_output(model, x, rand_index, masks)
        loss = self._calc_loss(output, t, rand_index, lams, self.criterion)
        retdict = dict(masks=masks, x_mix=x_mix, x_a=x_a, x_b=x_b)
        return loss, retdict

    def _sample_lambda(self, batch_size: int, beta_dist_a: float, beta_dist_b: float) -> torch.Tensor:
        assert batch_size > 0
        # for numpy.random.beta, please check docs: https://docs.scipy.org/doc/numpy-1.14.1/reference/generated/numpy.random.beta.html
        return torch.from_numpy(np.random.beta(beta_dist_a, beta_dist_b, size=batch_size)).float()


class Cutmix(Mixup):
    """
    implementation of CutMix (ICCV 2019): https://arxiv.org/abs/1905.04899
    """
    def __init__(self,
                 beta_dist_a: float = 1.0,
                 beta_dist_b: float = 1.0,
                 criterion=torch.nn.CrossEntropyLoss(),
                 device: str = 'cuda'):
        self.beta_dist_a = beta_dist_a
        self.beta_dist_b = beta_dist_b
        self.criterion = criterion
        self.device = device

    def __call__(self, model: torch.nn, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rand_index = torch.randperm(x.size(0))
        lams_rough = self._sample_lambda(x.size(0), self.beta_dist_a, self.beta_dist_b).to(self.device)
        masks = torch.stack([self.sample_square_mask(x.size(-1), lam.item())[None, :, :].repeat(3, 1, 1) for lam in lams_rough]).to(self.device)  # shape of masks is [b, 3, h, w]
        lams = masks[:, 0, :, :].sum(dim=(-1, -2)).to(self.device)  # shape of lams is [b]

        output, x_mix, x_a, x_b = self._clac_output(model, x, rand_index, masks)
        loss = self._calc_loss(output, t, rand_index, lams, self.criterion)
        retdict = dict(masks=masks, x_mix=x_mix, x_a=x_a, x_b=x_b)
        return loss, retdict

    @classmethod
    def sample_square_mask(cls, im_size: int, lam: float):
        assert im_size >= 2
        assert 0.0 <= lam <= 1.0

        window_size = int(im_size * np.sqrt(1.0 - lam))

        # sampling window center
        ch = np.random.randint(im_size)
        cw = np.random.randint(im_size)

        bbh_min = np.clip(ch - (window_size // 2), 0, im_size)
        bbw_min = np.clip(cw - (window_size // 2), 0, im_size)
        bbh_max = np.clip(ch + (window_size // 2), 0, im_size)
        bbw_max = np.clip(cw + (window_size // 2), 0, im_size)

        mask = torch.zeros(im_size, im_size, dtype=torch.float)
        mask[bbh_min:bbh_max, bbw_min:bbw_max] = torch.ones_like(mask, dtype=torch.float)[bbh_min:bbh_max, bbw_min:bbw_max]

        return mask


class SmoothMix(MixAugmentation):
    """
    implementation of SmoothMix (CVPR Workshop 2020): https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf
    """
    MASK_TYPE = ['circle', 'square']  # square mask_type is currently not supported.

    def __init__(self,
                 sigma: list,
                 criterion=torch.nn.CrossEntropyLoss(),
                 device: str = 'cuda'):
        self.sigma = sigma
        self.criterion = criterion
        self.device = device

    def __call__(self, model: torch.nn, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rand_index = torch.randperm(x.size(0))
        masks = self._sample_mask(x.size(0), x.size(-1), sigma=self.sigma).to(self.device)  # shape of masks is [b, 3, h, w]
        lams = masks[:, 0, :, :].sum(dim=(-1, -2)).to(self.device)  # shape of lams is [b]

        output, x_mix, x_a, x_b = self._clac_output(model, x, rand_index, masks)
        loss = self._calc_loss(output, t, rand_index, lams, self.criterion)
        retdict = dict(masks=masks, x_mix=x_mix, x_a=x_a, x_b=x_b)
        return loss, retdict

    def _sample_mask(self, batch_size: int, im_size: int, mask_type: str = 'circle', **kwargs):
        assert batch_size > 0
        assert im_size >= 2

        masks = list()
        if mask_type == 'circle':
            # kwargs should have 'sigma'
            assert 'sigma' in kwargs.keys(), 'mask_type == "circle" requires "sigma" as kwargs.'
            assert len(kwargs['sigma']) > 0
            sigmas = random.choices(kwargs['sigma'], k=batch_size)  # random sampling from list of sigma candidates.
            masks = [sample_gaussian_circle_mask(im_size, sigma)[None, :, :].repeat(3, 1, 1) for sigma in sigmas]
        else:
            raise NotImplementedError

        return torch.stack(masks, dim=0)


if __name__ == '__main__':
    import torchvision
    model = torchvision.models.resnet50(num_classes=10).cuda()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # model = torchvision.models.resnet50(num_classes=1000).cuda()
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.ToTensor()
    # ])
    # dataset = torchvision.datasets.ImageFolder(root='/media/gatheluck/gathe-drive/datasets/ImageNet/val', transform=transform)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    augmentors = {
        'mixup': Mixup(),
        'cutmix': Cutmix(),
        'smoothmix': SmoothMix(sigma=[8, 16])
    }

    for augmentor_name, augmentor in augmentors.items():
        for (x, t) in loader:
            x, t = x.cuda(), t.cuda()
            loss, retdict = augmentor(model, x, t)
            for k, v in retdict.items():
                torchvision.utils.save_image(v, '../logs/{augmentor}_{k}.png'.format(augmentor=augmentor_name, k=k))
            break
