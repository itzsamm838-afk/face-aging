"""Age a face with a trained model.

    python age_face.py --ckpt outputs/run1/ckpt_last.pt --image my_face.jpg --gender 0

Output: one row per input image -> [input, 0-5, 6-10, 11-15, 16-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71+]

NOTE: the model was trained on UTKFace's aligned & cropped faces (face fills the frame).
Photos that are not tightly cropped around the face will give poor results.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from models import Encoder, Generator
from utils import age_progression, pick_device, save_grid


def load_image(path, size):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)                                   # centre-crop to a square, then resize
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 127.5 - 1.0
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True, help="an image file, or a folder of images")
    p.add_argument("--gender", type=int, choices=[0, 1], required=True, help="0 = male, 1 = female")
    p.add_argument("--out", default="outputs/age_result.png")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    dev = pick_device(a.device)
    ck = torch.load(a.ckpt, map_location=dev)
    ta = ck["args"]
    E = Encoder(ta["img_size"], ta["z_dim"]).to(dev)
    G = Generator(ta["img_size"], ta["z_dim"], label_repeat=ta["label_repeat"]).to(dev)
    E.load_state_dict(ck["E"]); G.load_state_dict(ck["G"])
    E.eval(); G.eval()

    if os.path.isdir(a.image):
        paths = sorted(os.path.join(a.image, f) for f in os.listdir(a.image)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))[:16]
    else:
        paths = [a.image]
    if not paths:
        raise FileNotFoundError("No images found at " + a.image)

    x = torch.stack([load_image(q, ta["img_size"]) for q in paths]).to(dev)
    gender = torch.full((x.size(0),), a.gender, dtype=torch.long, device=dev)
    out = age_progression(E, G, x, gender)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_grid(out, 11, a.out)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
