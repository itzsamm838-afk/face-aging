"""Model definitions for a simplified Conditional Adversarial Autoencoder (CAAE).

Components
----------
Encoder E          : image x            -> latent z  (z_dim, tanh -> range [-1, 1])
Generator G        : (z, label)         -> image     (decoder, conditioned on age group + gender)
DiscriminatorZ Dz  : latent vector      -> real/fake logit  (real = sample from uniform prior)
DiscriminatorImg Di: (image, label)     -> real/fake logit  (real = true photo)

Label = one-hot age group (10 dims) + one-hot gender (2 dims) = 12 dims.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_AGE = 10      # age groups: 0-5, 6-10, 11-15, 16-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71+
N_GENDER = 2    # 0 = male, 1 = female (UTKFace convention)
N_LABEL = N_AGE + N_GENDER


def make_label(age_idx, gender):
    """age_idx, gender: LongTensor [B]  ->  float label [B, 12]."""
    return torch.cat(
        [F.one_hot(age_idx, N_AGE).float(), F.one_hot(gender, N_GENDER).float()], dim=1
    )


def _num_down(img_size):
    assert img_size >= 32 and (img_size & (img_size - 1)) == 0, \
        "img_size must be a power of 2 and >= 32 (e.g. 32, 64, 128)"
    return int(math.log2(img_size // 4))  # how many stride-2 steps to reach a 4x4 map


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    def __init__(self, img_size=64, z_dim=50, base=64):
        super().__init__()
        n = _num_down(img_size)
        layers, c_in, c = [], 3, base
        for _ in range(n):
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, inplace=True)]
            c_in, c = c, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in * 4 * 4, z_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return torch.tanh(self.fc(h))  # tanh so z lives in [-1, 1], matching the uniform prior


class Generator(nn.Module):
    def __init__(self, img_size=64, z_dim=50, base=64, label_repeat=10):
        super().__init__()
        n = _num_down(img_size)
        self.label_repeat = label_repeat
        c0 = min(base * 2 ** (n - 1), 512)
        self.c0 = c0
        self.fc = nn.Sequential(
            nn.Linear(z_dim + N_LABEL * label_repeat, c0 * 4 * 4),
            nn.BatchNorm1d(c0 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        chs = [c0]
        for _ in range(n - 1):
            chs.append(max(chs[-1] // 2, base))
        layers = []
        for i in range(n - 1):
            layers += [nn.ConvTranspose2d(chs[i], chs[i + 1], 4, 2, 1),
                       nn.BatchNorm2d(chs[i + 1]), nn.ReLU(inplace=True)]
        layers += [nn.ConvTranspose2d(chs[-1], 3, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z, label):
        # The 12-dim label is tiled so it is not drowned out by the 50-dim z.
        h = torch.cat([z, label.repeat(1, self.label_repeat)], dim=1)
        h = self.fc(h).view(-1, self.c0, 4, 4)
        return self.net(h)


class DiscriminatorZ(nn.Module):
    """Pushes E(x) to look like samples from the uniform prior U(-1, 1)."""

    def __init__(self, z_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, z):
        return self.net(z)


class DiscriminatorImg(nn.Module):
    """Real photo vs generated face, conditioned on the label (label is concatenated as extra channels)."""

    def __init__(self, img_size=64, base=64):
        super().__init__()
        n = _num_down(img_size)
        layers, c_in, c = [], 3 + N_LABEL, base
        for i in range(n):
            layers.append(nn.Conv2d(c_in, c, 4, 2, 1))
            if i > 0:
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            c_in, c = c, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in * 4 * 4, 1)

    def forward(self, x, label):
        b, _, h, w = x.shape
        lab = label[:, :, None, None].expand(b, label.size(1), h, w)
        return self.fc(self.conv(torch.cat([x, lab], dim=1)).flatten(1))
