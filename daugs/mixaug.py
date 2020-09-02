import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import torch

from daugs.mask import sample_gaussian_circle_mask


class MixAugmentation():
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor, t: torch.Tensor):
        pass


class Mixup(MixAugmentation):
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
        lam = self._sample_lambda(x.size(0), self.beta_dist_a, self.beta_dist_b).to(self.device)

        # augment input
        x_aug = (lam[:, None, None, None] * x) + ((torch.ones_like(lam) - lam)[:, None, None, None] * x[rand_index])
        output = model(x_aug)

        t_a = t
        t_b = t[rand_index]

        loss = (lam * self.criterion(output, t_a)) + ((torch.ones_like(lam) - lam) * self.criterion(output, t_b))

        return loss, x_aug.detach()

    def _sample_lambda(self, batch_size: int, beta_dist_a: float, beta_dist_b: float) -> torch.Tensor:
        assert batch_size > 0
        # for numpy.random.beta, please check docs: https://docs.scipy.org/doc/numpy-1.14.1/reference/generated/numpy.random.beta.html
        return torch.from_numpy(np.random.beta(beta_dist_a, beta_dist_b, size=batch_size)).float()


class SmoothMix():
    MASK_TYPE = ['circle', 'square']

    def __init__(self,
                 sigma: float,
                 criterion=torch.nn.CrossEntropyLoss(),
                 device: str = 'cuda'):
        self.sigma = sigma
        self.criterion = criterion
        self.device = device

    def __call__(self, model: torch.nn, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rand_index = torch.randperm(x.size(0))

        masks = self._sample_mask(x.size(0), x.size(-1), sigma=self.sigma).to(self.device)  # shape of masks is [b, 3, h, w]
        lams = masks[:, 0, :, :].sum(dim=(-1, -2)).to(self.device)  # shape of lams is [b]
        torchvision.utils.save_image((masks * x), '../logs/smoothmix_masked_a.png')
        torchvision.utils.save_image(((torch.ones_like(masks) - masks) * x[rand_index]), '../logs/smoothmix_masked_b.png')

        # augment input
        x_aug = (masks * x) + ((torch.ones_like(masks) - masks) * x[rand_index])
        output = model(x_aug)

        t_a = t
        t_b = t[rand_index]

        loss = (lams * self.criterion(output, t_a)) + ((torch.ones_like(lams) - lams) * self.criterion(output, t_b))

        return loss, x_aug.detach()

    def _sample_mask(self, batch_size: int, im_size: int, mask_type: str = 'circle', **kwargs):
        assert batch_size > 0
        assert im_size >= 2

        masks = list()
        if mask_type == 'circle':
            # kwargs should have 'sigma'
            assert 'sigma' in kwargs.keys(), 'mask_type == "circle" requires "sigma" as kwargs.'
            sigma = kwargs['sigma']
            masks = [sample_gaussian_circle_mask(im_size, sigma)[None, :, :].repeat(3, 1, 1) for _ in range(batch_size)]
        else:
            raise NotImplementedError

        return torch.stack(masks, dim=0)


if __name__ == '__main__':
    import torchvision
    # model = torchvision.models.resnet50(num_classes=10).cuda()
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.CenterCrop(32),
    #     torchvision.transforms.ToTensor()
    # ])
    # dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', download=True, transform=transform)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model = torchvision.models.resnet50(num_classes=1000).cuda()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder(root='/media/gatheluck/gathe-drive/datasets/ImageNet/val', transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    augmentors = {
        'mixup': Mixup(),
        'smoothmix': SmoothMix(sigma=32)
    }

    for k, augmentor in augmentors.items():
        for (x, t) in loader:
            x, t = x.cuda(), t.cuda()
            loss, x_aug = augmentor(model, x, t)
            torchvision.utils.save_image(x, '../logs/cifar10.png')
            torchvision.utils.save_image(x_aug, '../logs/{}.png'.format(k))
            break
