# Face Aging with a Conditional Adversarial Autoencoder (CAAE) - PyTorch

A small, readable PyTorch version of the idea from *"Age Progression/Regression by Conditional
Adversarial Autoencoder"* (Zhang, Song, Qi - CVPR 2017). You give it a face, it generates that
face at 10 age groups.

**Honest scope - read this first**
- This is a **simplified re-implementation of the idea**, not an exact copy of the paper or of the
  `ZZUTK/Face-Aging-CAAE` repo (that one is Python 2.7 + TensorFlow 1.7, which is very hard to run today).
- It was **tested end to end on synthetic data only** (loading, training, resume, sampling, inference,
  64px and 128px). It has **not** been trained on real UTKFace yet, so I cannot promise the face quality.
  The loss weights are sensible starting points, not tuned values. Expect to look at the sample images
  and adjust.
- Your job is to run it on real data, look at the results, and be able to explain it (see the last section).

---

## 1. Folder structure

```
face-aging-caae/
  models.py          Encoder, Generator, two discriminators
  dataset.py         reads UTKFace, turns file names into age-group + gender labels
  utils.py           helpers (image grids, TV loss, age_progression)
  train.py           training loop
  age_face.py        age your own face image with a trained model
  make_fake_data.py  makes fake data so you can test the setup before downloading anything
  plot_losses.py     draws loss curves from outputs/<run>/losses.csv
  data/              put the dataset here
  outputs/           training results go here
```

## 2. Setup

Python 3.9 or newer.

```bash
cd face-aging-caae
python -m venv venv
# Windows:  venv\Scripts\activate        Mac/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the matching PyTorch build from https://pytorch.org (it gives you the
exact command for your CUDA version). Then check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

No GPU? Use Google Colab (Section 6). Training on CPU works but is slow.

## 3. Step 0 - test the setup with fake data (2 minutes)

```bash
python make_fake_data.py --out data/fake_utk --n 200
python train.py --data_dir data/fake_utk --out_dir outputs/smoke --epochs 2 --batch_size 32
```

If this finishes and `outputs/smoke/samples/` has PNG files, your Python environment is fine.
The images are random colour blobs, so the results mean nothing.

## 4. Get the real dataset (UTKFace, aligned & cropped)

UTKFace has 20,000+ face images with age, gender and ethnicity labels. We use the **aligned and cropped**
version (faces fill the frame), which is what the original CAAE repo uses.

Where to download:
1. **Official page:** https://susanqq.github.io/UTKFace/ - the original repo's README points here.
2. **Hugging Face mirror:** https://huggingface.co/datasets/py97/UTKFace-Cropped - its README says the original
   aligned-and-cropped link is no longer available, so the uploader mirrored it. Use this if the official
   Google Drive link is dead (it is the most likely thing to break). To download with Python:
   ```python
   # pip install huggingface_hub
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="py97/UTKFace-Cropped", repo_type="dataset", local_dir="data/hf_utk")
   ```
   Or just use the "Files" tab on that page in your browser.

If you get a `.tar.gz` / `.zip`, extract it anywhere inside `data/`. **The folder layout does not matter**
because the loader searches sub-folders recursively. A good final result looks like
`data/UTKFace/25_0_1_20170116174525125.jpg.chip.jpg`.

File name format: `[age]_[gender]_[race]_[date&time].jpg`, with gender `0 = male`, `1 = female`.
We only use age and gender. Files with broken names are skipped automatically and counted in the log.

**Licence/ethics:** the dataset is for **non-commercial research use**. It contains real people's faces, so
do not use it for anything commercial, and do not run other people's photos through the model without
their permission.

## 5. Train

```bash
python train.py --data_dir data/UTKFace --out_dir outputs/run1
```

On start it prints how many images it loaded and **how many images per age group** - look at this table,
UTKFace is heavily imbalanced (see interview question 6).

Useful options:

| Option | Default | What it does |
|---|---|---|
| `--epochs` | 50 | number of passes over the data |
| `--img_size` | 64 | 64 is fast. 128 is the size the original repo used, ~4x slower |
| `--batch_size` | 64 | lower it if you run out of GPU memory |
| `--max_images` | all | random subset, for quick experiments |
| `--lambda_img`, `--lambda_z` | 0.01 | weight of the two adversarial losses vs reconstruction |
| `--resume path/to/ckpt_last.pt` | none | continue an interrupted run |

**Time it before committing:** run `--epochs 1` first and read the "seconds" at the end of the epoch line.
(For reference, the original TensorFlow repo reports about 2.5 hours for 50 epochs at 128px on a TITAN X.
I have not measured this PyTorch version on real data or a GPU.)

What you get in `outputs/run1/`:
- `samples/recon_XXX.png` - top row: held-out real faces, bottom row: reconstructions. **Check this first.**
  If reconstructions are not recognisable faces, nothing else will work.
- `samples/aging_XXX.png` - each row is one held-out face: `[input, 0-5, 6-10, 11-15, 16-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71+]`
- `losses.csv` - loss per epoch. Plot it with `python plot_losses.py outputs/run1`
- `ckpt_last.pt` - latest model (plus `ckpt_epoch_010.pt`, etc.)

## 6. No GPU? Google Colab

1. Zip this folder, upload it to Google Drive, open a new Colab notebook, set **Runtime -> Change runtime type -> GPU**.
2. Run in cells:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   %cd /content/drive/MyDrive/face-aging-caae
   !pip install -q pillow matplotlib
   !python train.py --data_dir data/UTKFace --out_dir outputs/run1 --epochs 50
   ```
   (Colab already has PyTorch.) Colab disconnects after a while - that is what `--resume` and the per-epoch
   `ckpt_last.pt` are for. Keep `outputs/` on Drive.

## 7. Age your own face

```bash
python age_face.py --ckpt outputs/run1/ckpt_last.pt --image my_face.jpg --gender 0
```
`--gender 0` = male, `1` = female. Output: `outputs/age_result.png`.
The model only knows tightly cropped, aligned faces like UTKFace. A full selfie with background will look bad.

## 8. How it works (short)

```
x --Encoder--> z (50 numbers, tanh)                  label = age group (10) + gender (2)
                |                                             |
                +--------------> Generator(z, label) <--------+----> reconstructed face x'

Dz : looks at z      -> is it from the uniform prior U(-1,1) or from the Encoder?   (shapes the latent space)
Di : looks at (image, label) -> is it a real photo or a generated one?              (makes outputs look like faces)
```
Loss for Encoder+Generator = `L1(x, x')` + `0.01 * (fool Dz)` + `0.01 * (fool Di)` + `1e-4 * TV`.
Aging at test time: encode the face once to get `z`, then decode the **same z** with each of the 10 age labels.

Deliberate simplifications vs the paper: L1 instead of L2 reconstruction; label concatenated to the
Di input instead of injected at a later layer; 64px by default; no extra tricks. These are things you can
mention as "what I would change / what I changed".

## 9. Questions you should be able to answer (interview prep)

1. Why is this an *autoencoder* and not a plain GAN? What does the encoder give you that a GAN generator alone doesn't?
2. What does `Dz` do, and why does `z` need a prior? What would happen to aging without it?
3. How does the age label change the output when `z` stays the same? Where does identity live, and where does age live?
4. Why is the reconstruction loss needed? What would the faces look like with only the adversarial losses?
5. Why are `lambda_img` and `lambda_z` small (0.01)? What do you expect if you set them to 1?
6. UTKFace has far fewer children and very old people than 20-40 year olds. How does that affect results, and how would you check it? (Look at the per-age-group table the loader prints.)
7. What does `Di` see in `x_rec` vs real `x`, and why do we `.detach()` `x_rec` in the Di step?
8. Reconstruction looks good but aging looks weak (all ages look the same). Give two possible causes and one experiment for each.
9. How would you *measure* aging quality, since "looks right" is subjective? (Hint: an age-estimation model, identity similarity.)
10. Why are eval-mode BatchNorm and train-mode BatchNorm different when sampling?

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `No images found under 'data/UTKFace'` | wrong `--data_dir`, or you didn't extract the archive |
| `none had UTKFace-style names` | file names are not `age_gender_race_date.jpg`; use the aligned & cropped UTKFace files |
| CUDA out of memory | lower `--batch_size` (32 or 16) or use `--img_size 64` |
| Very slow | you are on CPU. Use Colab, or add `--max_images 5000` for a quick experiment |
| Reconstructions are grey/blurry blobs after many epochs | check `recon_XXX.png` early; try `--lambda_img 0.001`; make sure more than a few thousand images loaded |
| Faces look good but identical across ages | increase `--label_repeat` (e.g. 20), or train longer |
| Colab session died | rerun with `--resume outputs/run1/ckpt_last.pt` |
