"""gen_soft_labels.py

Run the Cerberus teacher (resnet34) on YOUR OWN unlabeled histology images
(tiles or whole-slide images) and save the teacher's LOGITS for every output
head. These soft labels are later used to train the lightweight student via
knowledge distillation (see distill_train.py).

IMPORTANT: distillation does NOT need any manual annotation. We simply let the
teacher "teach" the student on unlabeled patches.

Usage examples
--------------
1) Your data is a folder of tile images (.png / .jpg):

   python distill/gen_soft_labels.py \
       --teacher_dir resnet34_cerberus \
       --input_dir /path/to/your/tiles \
       --input_type tile \
       --output_dir soft_labels

2) Your data is a folder of whole-slide images (.svs / .ndpi / .tif):

   python distill/gen_soft_labels.py \
       --teacher_dir resnet34_cerberus \
       --input_dir /path/to/your/wsis \
       --input_type wsi \
       --nr_tiles_per_wsi 200 \
       --output_dir soft_labels

Output
------
One ``<name>.pt`` file per input image inside ``--output_dir``. Each file
contains the 448x448 image tiles and the teacher logits for all heads
(Gland-INST, Lumen-INST, Nuclei-INST, Nuclei-TYPE, Patch-Class).
"""

import argparse
import os
import random
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import tqdm

CERBERUS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.insert(0, CERBERUS_ROOT)

from models.net_desc import create_model  # noqa: E402


# ----------------------------------------------------------------------------
# teacher loading
# ----------------------------------------------------------------------------
def load_teacher(teacher_dir, gpu):
    settings_path = os.path.join(teacher_dir, "settings.yml")
    weights_path = os.path.join(teacher_dir, "weights.tar")
    assert os.path.exists(settings_path), "settings.yml not found in %s" % teacher_dir
    assert os.path.exists(weights_path), "weights.tar not found in %s" % teacher_dir

    import yaml

    with open(settings_path) as fptr:
        run_paramset = yaml.full_load(fptr)
    model_args = run_paramset["model_kwargs"]

    net = create_model(**model_args)
    ckpt = torch.load(weights_path, map_location="cpu")
    saved_state_dict = ckpt["desc"]
    is_parallel = all(k.split(".")[0] == "module" for k in saved_state_dict.keys())
    if is_parallel:
        saved_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in saved_state_dict.items()
        }
    net.load_state_dict(saved_state_dict, strict=True)
    net = net.to("cuda:%d" % gpu).eval()
    for p in net.parameters():
        p.requires_grad = False
    print("Teacher loaded from %s" % weights_path)
    return net


# ----------------------------------------------------------------------------
# tile extraction helpers
# ----------------------------------------------------------------------------
def has_tissue(img, min_foreground):
    """Return True if the image is not mostly empty background."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tissue_frac = float((gray < 220).mean())
    return tissue_frac > min_foreground


def extract_tiles_from_image(img, tile_size, max_tiles):
    """Randomly sample `tile_size`x`tile_size` squares from a full image.

    Images larger than `tile_size` are randomly cropped.
    Images smaller than `tile_size` but still large enough for the network
    (>= 144 px, so the stride-16 bottom feature map is at least 9x9) are
    resized up to `tile_size` and used as a single tile.
    Images below that are skipped.
    """
    h, w = img.shape[:2]
    if h < tile_size or w < tile_size:
        if max(h, w) >= 144:
            return [cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)]
        return []
    tiles = []
    seen = set()
    attempts = 0
    while len(tiles) < max_tiles and attempts < max_tiles * 20:
        attempts += 1
        y0 = random.randint(0, h - tile_size)
        x0 = random.randint(0, w - tile_size)
        key = (y0 // 224, x0 // 224)
        if key in seen:
            continue
        seen.add(key)
        tiles.append(img[y0 : y0 + tile_size, x0 : x0 + tile_size])
    return tiles


def build_wsi_sampler(wsi_path, target_mpp, tile_size):
    """Return a function that samples random 448x448 RGB patches from a WSI.

    The sampling level is chosen to approximate `target_mpp` (0.5 um/px is the
    magnification the teacher was trained at, i.e. ~20x).
    """
    import openslide

    osr = openslide.OpenSlide(wsi_path)
    try:
        src_mpp = float(osr.properties.get("openslide.mpp-x", 0.0))
    except (TypeError, ValueError):
        src_mpp = 0.0

    level = 0
    downsample = 1.0
    if src_mpp > 0:
        best_err = 1e9
        for l in range(osr.level_count):
            d = osr.level_downsamples[l]
            err = abs(src_mpp * d - target_mpp)
            if err < best_err:
                best_err = err
                level = l
                downsample = d

    lvl_w, lvl_h = osr.level_dimensions[level]
    max_x0 = max(0, int(lvl_w * downsample - tile_size * downsample))
    max_y0 = max(0, int(lvl_h * downsample - tile_size * downsample))

    def sample(nr):
        patches = []
        for _ in range(nr):
            x0 = random.randint(0, max_x0)
            y0 = random.randint(0, max_y0)
            patch = osr.read_region(
                (x0, y0), level, (tile_size, tile_size), as_array=True
            )
            patch = patch[:, :, :3]  # drop alpha
            patches.append(patch)
        return patches

    print("  WSI %s -> level %d (downsample %.2fx)" % (os.path.basename(wsi_path), level, downsample))
    return sample


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_dir", required=True, help="folder with settings.yml + weights.tar (resnet34_cerberus)")
    parser.add_argument("--input_dir", required=True, help="folder of tiles OR WSIs")
    parser.add_argument("--input_type", choices=["tile", "wsi"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tile_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nr_tiles_per_wsi", type=int, default=100,
                        help="random patches to sample from each WSI")
    parser.add_argument("--max_tiles_per_img", type=int, default=48,
                        help="max random tiles to take from a large tile image")
    parser.add_argument("--target_mpp", type=float, default=0.5,
                        help="sampling resolution for WSIs (um/px), ~0.5 = 20x")
    parser.add_argument("--min_foreground", type=float, default=0.05,
                        help="skip patches with less tissue than this fraction")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    teacher = load_teacher(args.teacher_dir, args.gpu)
    tile_size = args.tile_size

    if args.input_type == "tile":
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        file_list = sorted(
            f for f in os.listdir(args.input_dir)
            if f.lower().endswith(exts)
        )
    else:
        exts = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs")
        file_list = sorted(
            f for f in os.listdir(args.input_dir)
            if f.lower().endswith(exts)
        )
    assert len(file_list) > 0, "No images found in %s" % args.input_dir

    pbar = tqdm.tqdm(file_list, desc="Generating soft labels", ascii=True)
    for fname in pbar:
        fpath = os.path.join(args.input_dir, fname)
        name = os.path.splitext(fname)[0]
        save_path = os.path.join(args.output_dir, name + ".pt")
        if os.path.exists(save_path):
            continue

        if args.input_type == "tile":
            img = cv2.imread(fpath)
            if img is None:
                print("  cannot read %s, skipping" % fpath)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tiles = extract_tiles_from_image(img, tile_size, args.max_tiles_per_img)
        else:
            sampler = build_wsi_sampler(fpath, args.target_mpp, tile_size)
            tiles = sampler(args.nr_tiles_per_wsi)

        if len(tiles) == 0:
            print("  %s: no usable tiles (image too small?)" % name)
            continue

        # keep only patches with tissue
        kept = [t for t in tiles if has_tissue(t, args.min_foreground)]
        if len(kept) == 0:
            print("  %s: all patches are background, skipping" % name)
            continue
        tiles = kept

        imgs = np.stack(tiles)  # (N, 448, 448, 3) uint8
        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to("cuda:%d" % args.gpu)

        all_logits = {}
        with torch.no_grad():
            for b0 in range(0, imgs_t.shape[0], args.batch_size):
                batch = imgs_t[b0 : b0 + args.batch_size]
                pred = teacher(batch)
                for head, logits in pred.items():
                    all_logits.setdefault(head, []).append(logits.detach().cpu().half())

        logits = {
            head: torch.cat(lst, 0) for head, lst in all_logits.items()
        }
        torch.save(
            {"name": name, "imgs": torch.from_numpy(imgs), "logits": logits},
            save_path,
        )
        pbar.set_postfix(tiles=len(imgs))

    print("Done. Soft labels saved in %s" % args.output_dir)


if __name__ == "__main__":
    main()