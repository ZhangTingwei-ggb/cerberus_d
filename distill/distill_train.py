"""distill_train.py

Knowledge-distillation training: a lightweight MobileNetV3 student learns to
reproduce the outputs of the Cerberus resnet34 teacher on UNLABELED histology
patches.

Step 1: run `gen_soft_labels.py` to create the teacher's soft labels.
Step 2: run this script to train the student.
Step 3: run `run_infer_tile.py` with the folder this script created.

Usage
-----
python distill/distill_train.py \
    --teacher_dir resnet34_cerberus \
    --soft_dir soft_labels \
    --output_dir mobilenet_v3_large_cerberus \
    --backbone mobilenet_v3_large \
    --epochs 20

The output folder (e.g. mobilenet_v3_large_cerberus/) contains `weights.tar`
and `settings.yml`, ready for `run_infer_tile.py`.
"""

import argparse
import collections
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

CERBERUS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.insert(0, CERBERUS_ROOT)

from models.net_desc import create_model  # noqa: E402


# default per-head loss weights (mirror the training schedule of the teacher)
HEAD_WEIGHTS = {
    "Gland-INST": 1.4,
    "Gland-TYPE": 1.0,
    "Lumen-INST": 1.5,
    "Nuclei-INST": 1.0,
    "Nuclei-TYPE": 1.0,
    "Patch-Class": 0.4,
}


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def load_teacher(teacher_dir, gpu):
    settings_path = os.path.join(teacher_dir, "settings.yml")
    weights_path = os.path.join(teacher_dir, "weights.tar")
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
    return net, run_paramset


def build_student(backbone_name, decoder_kwargs, considered_tasks, pretrained, gpu,
                  init_weights=None):
    net = create_model(
        encoder_backbone_name=backbone_name,
        backbone_imagenet_pretrained=pretrained,
        fullnet_custom_pretrained=(init_weights is not None),
        decoder_kwargs=decoder_kwargs,
        considered_tasks=considered_tasks,
    )
    if init_weights is not None:
        ckpt = torch.load(init_weights, map_location="cpu")
        sd = ckpt["desc"]
        is_parallel = all(k.split(".")[0] == "module" for k in sd.keys())
        if is_parallel:
            sd = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        print("Student initialized from %s" % init_weights)
    net = net.to("cuda:%d" % gpu)
    nr_params = sum(p.numel() for p in net.parameters())
    print("Student %s params: %.2fM" % (backbone_name, nr_params / 1e6))
    return net


def configure_grads(student, patch_only):
    """In patch-only stage 2, freeze everything except backbone + Patch-Class head."""
    for name, p in student.named_parameters():
        p.requires_grad = True
    if patch_only:
        for name, p in student.named_parameters():
            if (
                name.startswith("backbone.")
                or name.startswith("decoder_head.Patch-Class.")
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False


def apply_mode(student, patch_only):
    """Put model in train mode, but force frozen submodules into eval mode so
    their BatchNorm running statistics do not get updated during stage 2."""
    student.train()
    if not patch_only:
        return
    for name, module in student.named_modules():
        if name == "" or name == "decoder_head":
            continue
        if (
            name.startswith("backbone")
            or name.startswith("decoder_head.Patch-Class")
            or name == "global_avg_pool"
        ):
            continue
        module.eval()


# ----------------------------------------------------------------------------
# dataset (soft-label .pt files on disk, LRU cached)
# ----------------------------------------------------------------------------
class LRUCache(collections.OrderedDict):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


def _augment(img, logits):
    """Apply the SAME random flip/rotation to image (H,W,3) and logits (C,H,W)."""
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k)
        logits = {h: np.rot90(v, k, axes=(1, 2)) for h, v in logits.items()}
    if random.random() < 0.5:
        img = img[:, ::-1]
        logits = {h: v[:, :, ::-1] for h, v in logits.items()}
    if random.random() < 0.5:
        img = img[::-1]
        logits = {h: v[:, ::-1] for h, v in logits.items()}
    return img, logits


def make_batch(data, heads, nr_tiles):
    """Pick `nr_tiles` random tiles from one loaded .pt file, augmented."""
    n = data["imgs"].shape[0]
    nr_tiles = min(nr_tiles, n)
    idxs = random.sample(range(n), nr_tiles)
    if len(idxs) == 1:
        idxs = idxs * 2  # BatchNorm needs batch>=2; duplicate (aug differs per copy)
    img_list = []
    logit_list = {h: [] for h in heads}
    for i in idxs:
        img = data["imgs"][i].numpy().astype(np.float32)
        logits = {h: data["logits"][h][i].numpy() for h in heads}
        img, logits = _augment(img, logits)
        img_list.append(img)
        for h in heads:
            logit_list[h].append(torch.from_numpy(np.copy(logits[h])))
    imgs = torch.from_numpy(np.stack(img_list))  # (K,448,448,3)
    imgs = imgs.permute(0, 3, 1, 2).contiguous()  # (K,3,448,448) NCHW
    logits = {h: torch.stack(logit_list[h]) for h in heads}  # (K,C,...)
    return imgs, logits


# ----------------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------------
def kd_loss(student_logits, teacher_logits, temperature):
    """KL divergence between teacher & student soft targets (per pixel/class)."""
    s = F.log_softmax(student_logits / temperature, dim=1)
    t = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(s, t, reduction="mean") * (temperature ** 2)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--soft_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", default="mobilenet_v3_large",
                        choices=["mobilenet_v3_large", "mobilenet_v3_small", "mobilenet_v2"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--files_per_step", type=int, default=4,
                        help="soft-label files used per optimizer step")
    parser.add_argument("--tiles_per_file", type=int, default=4,
                        help="random tiles sampled from each file per step")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="do not start from ImageNet pretrained backbone")
    parser.add_argument("--patch_only", action="store_true",
                        help="stage 2: freeze backbone-grad except Patch-Class head, "
                             "only fine-tune the Patch-Class task on tissue-type soft labels")
    parser.add_argument("--init_weights", default=None,
                        help="path to a student weights.tar to continue from (stage 1 -> stage 2)")
    parser.add_argument("--cache_size", type=int, default=8,
                        help="number of soft-label files cached in RAM")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- models ---
    teacher, run_paramset = load_teacher(args.teacher_dir, args.gpu)
    teacher_model_args = run_paramset["model_kwargs"]
    decoder_kwargs = teacher_model_args["decoder_kwargs"]
    considered_tasks = teacher_model_args["considered_tasks"]
    student = build_student(
        args.backbone, decoder_kwargs, considered_tasks,
        pretrained=(not args.no_pretrained) and (args.init_weights is None),
        gpu=args.gpu, init_weights=args.init_weights,
    )
    if args.patch_only:
        configure_grads(student, patch_only=True)
        print("PATCH-ONLY mode: only the Patch-Class head is trainable")
    # which decoder heads get gradients (official train_decoder_list mechanism)
    decoder_keys = list(student.decoder_head.keys())
    train_decoder_list = ["Patch-Class"] if args.patch_only else decoder_keys

    # --- data ---
    file_list = sorted(
        f for f in os.listdir(args.soft_dir) if f.endswith(".pt")
    )
    assert len(file_list) > 0, "No soft-label .pt files in %s" % args.soft_dir
    file_list = [os.path.join(args.soft_dir, f) for f in file_list]
    random.shuffle(file_list)
    nr_val = max(1, int(len(file_list) * args.val_split))
    val_files = file_list[:nr_val]
    train_files = file_list[nr_val:]
    print("train files: %d, val files: %d" % (len(train_files), len(val_files)))

    heads = ["Patch-Class"] if args.patch_only else sorted(HEAD_WEIGHTS.keys())
    cache = LRUCache(args.cache_size)

    def get_data(fpath):
        data = cache.get(fpath)
        if data is None:
            data = torch.load(fpath, map_location="cpu")
            cache.put(fpath, data)
        return data

    # --- optimizer ---
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- training ---
    best_val = float("inf")
    step_count = 0
    for epoch in range(args.epochs):
        random.shuffle(train_files)
        apply_mode(student, args.patch_only)
        running = 0.0
        nr_step = 0
        for fpath in train_files:
            data = get_data(fpath)
            imgs, teacher_logits = make_batch(data, heads, args.tiles_per_file)
            imgs = imgs.to("cuda:%d" % args.gpu)

            with torch.no_grad():
                teacher_out = teacher(imgs)

            student_out = student(imgs, train_decoder_list=train_decoder_list)
            loss = 0.0
            for h in heads:
                loss = loss + HEAD_WEIGHTS[h] * kd_loss(
                    student_out[h], teacher_out[h], args.temperature
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_count += 1
            running += loss.item()
            nr_step += 1
            if step_count % 50 == 0:
                print("  epoch %d step %d | kd loss %.4f" % (epoch + 1, step_count, loss.item()))

        scheduler.step()

        # --- validation ---
        student.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for fpath in val_files:
                data = get_data(fpath)
                imgs, teacher_logits = make_batch(data, heads, args.tiles_per_file)
                imgs = imgs.to("cuda:%d" % args.gpu)
                teacher_out = teacher(imgs)
                student_out = student(imgs, train_decoder_list=train_decoder_list)
                loss = sum(
                    HEAD_WEIGHTS[h] * kd_loss(student_out[h], teacher_out[h], args.temperature)
                    for h in heads
                )
                val_loss += loss.item()
                val_n += 1
        val_loss /= max(1, val_n)
        print("epoch %d | train kd %.4f | val kd %.4f" % (epoch + 1, running / max(1, nr_step), val_loss))

        if val_loss < best_val:
            best_val = val_loss
            sd = {"module." + k: v for k, v in student.state_dict().items()}
            torch.save(
                {"desc": sd, "optimizer": optimizer.state_dict(),
                 "lr_scheduler": scheduler.state_dict()},
                os.path.join(args.output_dir, "weights.tar"),
            )
            # settings.yml compatible with run_infer_tile.py
            new_paramset = {k: v for k, v in run_paramset.items() if k != "model_kwargs"}
            new_model_args = dict(decoder_kwargs=decoder_kwargs,
                                  considered_tasks=considered_tasks,
                                  encoder_backbone_name=args.backbone,
                                  backbone_imagenet_pretrained=not args.no_pretrained,
                                  fullnet_custom_pretrained=False)
            new_paramset["model_kwargs"] = new_model_args
            with open(os.path.join(args.output_dir, "settings.yml"), "w") as fptr:
                yaml.safe_dump(new_paramset, fptr, sort_keys=False)
            print("  saved best student (val kd %.4f) -> %s" % (best_val, args.output_dir))

    print("Finished. Best validation KD loss: %.4f" % best_val)


if __name__ == "__main__":
    main()