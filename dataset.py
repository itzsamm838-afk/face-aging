"""UTKFace loader.

UTKFace file names look like:   [age]_[gender]_[race]_[date&time].jpg(.chip.jpg)
    age    : integer years
    gender : 0 = male, 1 = female
We only use age and gender. Files whose names don't parse (a few UTKFace files are
malformed, plus stray files like '._xxx' from macOS zips) are skipped.

All images are loaded once, resized, and kept in RAM as uint8 (fast training, no workers needed).
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

AGE_BINS = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30),
            (31, 40), (41, 50), (51, 60), (61, 70), (71, 200)]
AGE_NAMES = ["0-5", "6-10", "11-15", "16-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71+"]


def age_to_group(age):
    for i, (lo, hi) in enumerate(AGE_BINS):
        if lo <= age <= hi:
            return i
    return None


def parse_utkface_name(path):
    parts = os.path.basename(path).split("_")
    if len(parts) < 4:
        return None
    try:
        age, gender = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if gender not in (0, 1):
        return None
    g = age_to_group(age)
    if g is None:
        return None
    return g, gender


class UTKFaceDataset(Dataset):
    def __init__(self, root, img_size=64, max_images=None, flip=True, seed=0):
        exts = (".jpg", ".jpeg", ".png")
        files = []
        for dp, _, fns in os.walk(root):          # recursive: folder layout doesn't matter
            for fn in fns:
                if fn.lower().endswith(exts):
                    files.append(os.path.join(dp, fn))
        if not files:
            raise FileNotFoundError(
                f"No images found under '{root}'. Put the UTKFace .jpg files there (see README).")
        files.sort()
        random.Random(seed).shuffle(files)         # so --max_images gives a random subset, not just babies

        imgs, ages, genders, skipped = [], [], [], 0
        for p in files:
            if max_images and len(imgs) >= max_images:
                break
            parsed = parse_utkface_name(p)
            if parsed is None:
                skipped += 1
                continue
            try:
                im = Image.open(p).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
            except Exception:
                skipped += 1
                continue
            imgs.append(np.asarray(im))
            ages.append(parsed[0])
            genders.append(parsed[1])
            if len(imgs) % 5000 == 0:
                print(f"  loaded {len(imgs)} images ...")

        if not imgs:
            raise RuntimeError("Found image files, but none had UTKFace-style names "
                               "([age]_[gender]_[race]_[date].jpg).")
        self.images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,H,W
        self.age = torch.tensor(ages, dtype=torch.long)
        self.gender = torch.tensor(genders, dtype=torch.long)
        self.flip = flip
        print(f"Loaded {len(self)} images ({skipped} skipped) at {img_size}x{img_size}")
        counts = torch.bincount(self.age, minlength=len(AGE_NAMES)).tolist()
        print("Images per age group:", dict(zip(AGE_NAMES, counts)))
        print(f"Gender counts (male, female): {torch.bincount(self.gender, minlength=2).tolist()}")

    def __len__(self):
        return self.images.size(0)

    def get(self, i, flip=False):
        x = self.images[i].float() / 127.5 - 1.0   # [-1, 1], matches the generator's tanh output
        if flip and random.random() < 0.5:
            x = x.flip(-1)
        return x, self.age[i], self.gender[i]

    def __getitem__(self, i):
        return self.get(i, self.flip)
