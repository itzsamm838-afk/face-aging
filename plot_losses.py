"""python plot_losses.py outputs/run1   ->  outputs/run1/losses.png"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1] if len(sys.argv) > 1 else "outputs/run1"
with open(os.path.join(run, "losses.csv")) as f:
    rows = list(csv.DictReader(f))
ep = [int(r["epoch"]) for r in rows]

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(ep, [float(r["recon_l1"]) for r in rows]); ax[0].set_title("Reconstruction L1 (lower = better)")
ax[1].plot(ep, [float(r["dz"]) for r in rows], label="Dz"); ax[1].plot(ep, [float(r["di"]) for r in rows], label="Di")
ax[1].set_title("Discriminator losses"); ax[1].legend()
ax[2].plot(ep, [float(r["eg_z"]) for r in rows], label="E fools Dz"); ax[2].plot(ep, [float(r["eg_img"]) for r in rows], label="G fools Di")
ax[2].set_title("Encoder/Generator adversarial losses"); ax[2].legend()
for a in ax:
    a.set_xlabel("epoch")
plt.tight_layout()
out = os.path.join(run, "losses.png")
plt.savefig(out, dpi=120)
print("Saved", out)
