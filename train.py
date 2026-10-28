"""Train the simplified CAAE on UTKFace.

Per batch:
  1. z = E(x);  x_rec = G(z, label)
  2. Dz  step : real = z ~ U(-1,1),  fake = E(x)
  3. Di  step : real = (x, label),   fake = (x_rec, label)
  4. E+G step : L1 reconstruction + lambda_z * (fool Dz) + lambda_img * (fool Di) + lambda_tv * TV
"""
import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import UTKFaceDataset
from models import (DiscriminatorImg, DiscriminatorZ, Encoder, Generator,
                    init_weights, make_label)
from utils import age_progression, pick_device, save_grid, set_seed, tv_loss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/UTKFace")
    p.add_argument("--out_dir", default="outputs/run1")
    p.add_argument("--img_size", type=int, default=64, help="64 is fast; 128 matches the original repo")
    p.add_argument("--z_dim", type=int, default=50)
    p.add_argument("--label_repeat", type=int, default=10)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_z", type=float, default=0.01)
    p.add_argument("--lambda_img", type=float, default=0.01)
    p.add_argument("--lambda_tv", type=float, default=1e-4)
    p.add_argument("--max_images", type=int, default=None, help="use a random subset (quick tests)")
    p.add_argument("--keep_every", type=int, default=10, help="also keep a numbered checkpoint every N epochs")
    p.add_argument("--resume", default=None, help="path to a checkpoint to continue from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def bce(logits, target):
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))


def main():
    args = get_args()
    set_seed(args.seed)
    dev = pick_device(args.device)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    print("Device:", dev)

    ds = UTKFaceDataset(args.data_dir, args.img_size, args.max_images)
    if len(ds) < 16:
        raise RuntimeError("Need at least 16 usable images.")
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_val = max(8, int(0.05 * len(ds)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)
    if len(loader) == 0:
        raise RuntimeError("batch_size is larger than the training set; lower --batch_size.")

    # Fixed held-out faces, used to visualise progress after every epoch.
    nv = min(8, len(val_idx))
    viz = [ds.get(i, flip=False) for i in val_idx[:nv]]
    viz_x = torch.stack([v[0] for v in viz]).to(dev)
    viz_gender = torch.stack([v[2] for v in viz]).to(dev)
    viz_age = torch.stack([v[1] for v in viz]).to(dev)

    E = Encoder(args.img_size, args.z_dim).to(dev)
    G = Generator(args.img_size, args.z_dim, label_repeat=args.label_repeat).to(dev)
    Dz = DiscriminatorZ(args.z_dim).to(dev)
    Di = DiscriminatorImg(args.img_size).to(dev)
    for m in (E, G, Dz, Di):
        m.apply(init_weights)

    betas = (0.5, 0.999)
    opt_eg = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=args.lr, betas=betas)
    opt_dz = torch.optim.Adam(Dz.parameters(), lr=args.lr, betas=betas)
    opt_di = torch.optim.Adam(Di.parameters(), lr=args.lr, betas=betas)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        E.load_state_dict(ck["E"]); G.load_state_dict(ck["G"])
        Dz.load_state_dict(ck["Dz"]); Di.load_state_dict(ck["Di"])
        opt_eg.load_state_dict(ck["opt_eg"]); opt_dz.load_state_dict(ck["opt_dz"]); opt_di.load_state_dict(ck["opt_di"])
        start_epoch = ck["epoch"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, "losses.csv")
    if not os.path.exists(log_path) or start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "recon_l1", "dz", "di", "eg_z", "eg_img", "seconds"])

    for epoch in range(start_epoch, args.epochs):
        for m in (E, G, Dz, Di):
            m.train()
        t0 = time.time()
        sums = dict(rec=0.0, dz=0.0, di=0.0, ez=0.0, gi=0.0)

        for x, age, gender in loader:
            x, age, gender = x.to(dev), age.to(dev), gender.to(dev)
            label = make_label(age, gender)
            b = x.size(0)

            z = E(x)
            x_rec = G(z, label)

            # --- Dz: prior samples are "real", encoder outputs are "fake"
            z_prior = torch.rand(b, args.z_dim, device=dev) * 2 - 1
            loss_dz = bce(Dz(z_prior), 1.0) + bce(Dz(z.detach()), 0.0)
            opt_dz.zero_grad(set_to_none=True)
            loss_dz.backward()
            opt_dz.step()

            # --- Di: real photos are "real", reconstructions are "fake"
            loss_di = bce(Di(x, label), 1.0) + bce(Di(x_rec.detach(), label), 0.0)
            opt_di.zero_grad(set_to_none=True)
            loss_di.backward()
            opt_di.step()

            # --- E + G
            loss_rec = F.l1_loss(x_rec, x)
            loss_ez = bce(Dz(z), 1.0)
            loss_gi = bce(Di(x_rec, label), 1.0)
            loss_eg = (args.lambda_recon * loss_rec + args.lambda_z * loss_ez
                       + args.lambda_img * loss_gi + args.lambda_tv * tv_loss(x_rec))
            opt_eg.zero_grad(set_to_none=True)
            loss_eg.backward()
            opt_eg.step()

            sums["rec"] += loss_rec.item(); sums["dz"] += loss_dz.item(); sums["di"] += loss_di.item()
            sums["ez"] += loss_ez.item(); sums["gi"] += loss_gi.item()

        n = len(loader)
        avg = {k: v / n for k, v in sums.items()}
        secs = time.time() - t0
        print(f"epoch {epoch + 1:03d}/{args.epochs} | recon L1 {avg['rec']:.4f} | Dz {avg['dz']:.3f} | "
              f"Di {avg['di']:.3f} | E(z) {avg['ez']:.3f} | G(img) {avg['gi']:.3f} | {secs:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg["rec"], avg["dz"], avg["di"], avg["ez"], avg["gi"], round(secs, 1)])

        # --- visualise on held-out faces
        for m in (E, G):
            m.eval()
        with torch.no_grad():
            rec = G(E(viz_x), make_label(viz_age, viz_gender))
            save_grid(torch.cat([viz_x, rec]), nv,
                      os.path.join(args.out_dir, "samples", f"recon_{epoch + 1:03d}.png"))
            save_grid(age_progression(E, G, viz_x, viz_gender), 11,
                      os.path.join(args.out_dir, "samples", f"aging_{epoch + 1:03d}.png"))

        ck = dict(E=E.state_dict(), G=G.state_dict(), Dz=Dz.state_dict(), Di=Di.state_dict(),
                  opt_eg=opt_eg.state_dict(), opt_dz=opt_dz.state_dict(), opt_di=opt_di.state_dict(),
                  epoch=epoch + 1, args=vars(args))
        torch.save(ck, os.path.join(args.out_dir, "ckpt_last.pt"))
        if (epoch + 1) % args.keep_every == 0:
            torch.save(ck, os.path.join(args.out_dir, f"ckpt_epoch_{epoch + 1:03d}.pt"))

    print("Done. Look in", os.path.join(args.out_dir, "samples"))


if __name__ == "__main__":
    main()
