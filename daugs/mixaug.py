import numpy as np
import torch


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


if __name__ == '__main__':
    import torchvision
    model = torchvision.models.resnet50(num_classes=10).cuda()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    augmentor = Mixup()

    for (x, t) in loader:
        x, t = x.cuda(), t.cuda()
        loss, x_aug = augmentor(model, x, t)
        torchvision.utils.save_image(x_aug, '../logs/mixup.png')
        raise NotImplementedError
