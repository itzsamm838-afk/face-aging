"""Create random blob images named like UTKFace files, so you can test the whole pipeline
(loading -> training -> sampling -> age_face.py) BEFORE downloading the real dataset.

    python make_fake_data.py --out data/fake_utk --n 300

These images are NOT faces. Results trained on them mean nothing; this only checks that your setup works.
"""
import argparse
import os
import random

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fake_utk")
    p.add_argument("--n", type=int, default=300)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(a.n):
        low = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)           # low-frequency colour blobs
        im = Image.fromarray(low).resize((200, 200), Image.BICUBIC)
        name = f"{random.randint(1, 100)}_{random.randint(0, 1)}_{random.randint(0, 4)}_2017{i:011d}.jpg.chip.jpg"
        im.save(os.path.join(a.out, name))
    print(f"Wrote {a.n} fake images to {a.out}")


if __name__ == "__main__":
    main()
