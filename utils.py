import random

import numpy as np
import torch
from PIL import Image

from models import N_AGE, make_label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tv_loss(x):
    """Total variation: penalises noisy pixel-to-pixel jumps (smoother outputs)."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def save_grid(t, per_row, path, pad=2):
    """t: [N,3,H,W] in [-1,1]  ->  PNG grid with `per_row` images per row."""
    t = ((t.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    n, _, h, w = t.shape
    rows = (n + per_row - 1) // per_row
    canvas = np.full((rows * (h + pad) + pad, per_row * (w + pad) + pad, 3), 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, per_row)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = t[i].permute(1, 2, 0).numpy()
    Image.fromarray(canvas).save(path)


@torch.no_grad()
def age_progression(E, G, x, gender):
    """For each input face, generate it at all 10 age groups (same identity code z, same gender).

    Returns [B*(1+10), 3, H, W]: each row = [input, age 0-5, 6-10, ..., 71+].
    """
    z = E(x)
    cols = [x]
    for k in range(N_AGE):
        age = torch.full((x.size(0),), k, dtype=torch.long, device=x.device)
        cols.append(G(z, make_label(age, gender)))
    return torch.stack(cols, dim=1).flatten(0, 1)
