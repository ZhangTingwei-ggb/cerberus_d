"""
最简蒸馏训练模板 v0：GT + Output KD (KL/CS 二选一)

对应你方案的 A2 阶段：
  L_total = L_gt(核+腺体+Patch*0.5) + λ * L_output_kd
  L_gt    含 Dice+CE (分割) + CE (分类)，Patch 0.5
  L_output 默认为 CS散度 T=4，也可用 KL T=4 (--kd_type kl/cs)

数据假设 (先跑通，不依赖完整 PanNuke/GlaS/Kather 标注流)：
  - 若你已有真GT (PanNuke/GlaS/Kather100K)，把 gt_dict 按 distill/losses.py 约定喂入
  - 若暂时没有真GT，可令 --no_gt，仅跑 Output KD (退化成 A1)
  - 本模板同时支持 D_l 真GT + D_u 管腔伪 的混合：管腔辅头默认关闭，需要时 --enable_lumen_pseudo

Student / Teacher:
  - Teacher: cerberus_d/checkpoint/resnet34_cerberus (冻结, eval)
  - Student: resnet18 / enet / mobilenet_v3_large (三选一)

与现有 distill_train.py 的区别：
  - 现有 distill_train.py 只做 Output KD 的软标签文件 (soft_labels/*.pt)，无 GT
  - 本模板在同一脚本内同时算 GT + Output KD，Teacher 在线前向产 logits (不需要预先 gen_soft_labels)
  - 若你想复用已生成的 soft_labels，也可把 teacher_out 换成从 .pt 文件加载

用法 (在 cerberus_d 根目录下):
  # 1) 纯 KL 基线 + GT (A2)
  python distill/distill_train_gt_output.py \
    --teacher_dir checkpoint/resnet34_cerberus \
    --backbone resnet18 --kd_type kl --temperature 4 \
    --gt_weight 1.0 --kd_weight 10.0 --patch_gt_weight 0.5 \
    --output_dir checkpoint/distill/resnet18_gt_kl_T4

  # 2) CS 散度 (A4 的 Output 部分)
  python distill/distill_train_gt_output.py \
    --teacher_dir checkpoint/resnet34_cerberus \
    --backbone enet --kd_type cs --temperature 4 \
    --gt_weight 1.0 --kd_weight 10.0 \
    --output_dir checkpoint/distill/enet_gt_cs_T4

  # 3) 三个学生一键对比 (GT+CS)
  for bb in resnet18 enet mobilenet_v3_large; do
    python distill/distill_train_gt_output.py \
      --teacher_dir checkpoint/resnet34_cerberus --backbone $bb --kd_type cs \
      --output_dir checkpoint/distill/${bb}_gt_cs_T4
  done

  # 4) 消融：无GT纯蒸馏 (A1)
  python distill/distill_train_gt_output.py --no_gt --kd_type kl ...

数据:
  本模板内置 DummyGTDataset 演示 GT 形状，真正训练时替换为你的 PanNuke/GlaS/Kather loader。
  Mixed Batch 8+8 需在 dataset 层实现，这里先用单一批次演示 loss 计算通路。
"""
import argparse
import os
import random
import warnings
warnings.filterwarnings("ignore")

import yaml
import torch
import torch.nn.functional as F

CERBERUS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, CERBERUS_ROOT)

from models.net_desc import create_model  # noqa: E402
from distill.losses import DistillLoss  # noqa: E402

# ---------------------------------------------------------------------------
# teacher / student
# ---------------------------------------------------------------------------
def load_teacher(teacher_dir, gpu):
    settings_path = os.path.join(teacher_dir, "settings.yml")
    weights_path = os.path.join(teacher_dir, "weights.tar")
    with open(settings_path) as f:
        run_paramset = yaml.safe_load(f)
    model_args = run_paramset["model_kwargs"]
    net = create_model(**model_args)
    ckpt = torch.load(weights_path, map_location="cpu")
    sd = ckpt["desc"]
    # 去掉 DataParallel 前缀
    is_parallel = all(k.startswith("module.") for k in sd.keys())
    if is_parallel:
        sd = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    net = net.to(f"cuda:{gpu}").eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, run_paramset


def build_student(backbone, decoder_kwargs, considered_tasks, pretrained, gpu):
    net = create_model(
        encoder_backbone_name=backbone,
        backbone_imagenet_pretrained=pretrained,
        fullnet_custom_pretrained=False,
        decoder_kwargs=decoder_kwargs,
        considered_tasks=considered_tasks,
    )
    net = net.to(f"cuda:{gpu}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Student {backbone} params: {n_params/1e6:.2f}M")
    return net


# ---------------------------------------------------------------------------
# dummy GT dataset — 仅用于验证 loss 通路，替换为真实 loader
# ---------------------------------------------------------------------------
class DummyGTDataset(torch.utils.data.Dataset):
    """产生随机图 + 随机GT，形状与 Cerberus 一致，用于跑通链路。"""
    def __init__(self, length=64, img_size=448):
        self.length = length
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        H = W = self.img_size
        img = torch.randint(0, 256, (3, H, W), dtype=torch.float32)  # 0-255, NetDesc内会/255
        gt = {
            "Nuclei-INST": torch.randint(0, 3, (H, W), dtype=torch.long),
            "Nuclei-TYPE": torch.randint(0, 7, (H, W), dtype=torch.long),
            "Gland-INST":  torch.randint(0, 3, (H, W), dtype=torch.long),
            "Gland-TYPE":  torch.randint(0, 3, (H, W), dtype=torch.long),
            "Patch-Class": torch.randint(0, 9, (), dtype=torch.long),
        }
        return img, gt


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    # gt: list of dict -> dict of stacked tensors
    keys = batch[0][1].keys()
    gt_dict = {}
    for k in keys:
        vals = [b[1][k] for b in batch]
        # Patch-Class 是标量 (B,) , 其余是 (H,W)
        gt_dict[k] = torch.stack(vals)
    return imgs, gt_dict


# ---------------------------------------------------------------------------
# main — 单机单卡最小可跑版本
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cerberus GT+Output KD 最简模板 (A2)")
    parser.add_argument("--teacher_dir", required=True, help="checkpoint/resnet34_cerberus")
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "enet", "mobilenet_v3_large", "mobilenet_v3_small", "mobilenet_v2"])
    parser.add_argument("--kd_type", default="cs", choices=["kl", "cs", "mse"],
                        help="Output KD 类型：kl=KL散度 cs=CS散度 mse=Sigmoid MSE")
    parser.add_argument("--temperature", type=float, default=4.0, help="KD 温度，分类 T=4 分割二值 T=1")
    parser.add_argument("--gt_weight", type=float, default=1.0, help="L_gt 总权重")
    parser.add_argument("--kd_weight", type=float, default=10.0, help="L_output 总权重 (方案里 10x)")
    parser.add_argument("--patch_gt_weight", type=float, default=0.5, help="Patch GT 0.5 防喧宾夺主")
    parser.add_argument("--no_gt", action="store_true", help="不算GT，纯KD (A1)")
    parser.add_argument("--enable_lumen_pseudo", action="store_true", help="开启管腔辅头伪蒸馏 0.3")
    parser.add_argument("--epochs", type=int, default=2, help="演示 epoch 数，真正训练设 50+")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_pretrained", action="store_true", help="student 不用 ImageNet 预训练")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=5)
    # --- 正式训练数据目录（不传则回落 Dummy）---
    parser.add_argument("--lizard_root", default="/home/linjiatai/zhang/cerberus_d/dataset/nuclear/lizard", help="lizard 根目录")
    parser.add_argument("--glas_root", default="/home/linjiatai/zhang/cerberus_d/dataset/gland/GlaS", help="GlaS 根目录")
    parser.add_argument("--kather_root", default="/home/linjiatai/zhang/cerberus_d/dataset/classification/Kather100K", help="Kather100K 根目录")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Backbone: {args.backbone}  KD: {args.kd_type} T={args.temperature}")

    # --- models ---
    teacher, run_paramset = load_teacher(args.teacher_dir, args.gpu if torch.cuda.is_available() else 0)
    t_args = run_paramset["model_kwargs"]
    decoder_kwargs = t_args["decoder_kwargs"]
    considered_tasks = t_args["considered_tasks"]

    student = build_student(args.backbone, decoder_kwargs, considered_tasks,
                            pretrained=(not args.no_pretrained), gpu=args.gpu if torch.cuda.is_available() else 0)
    train_decoder_list = list(student.decoder_head.keys())

    # --- loss ---
    # 按 kd_type 覆盖 OutputKDLoss 的默认 head_cfg
    # v0: 全部用 cs T4（含 Nuclei-INST），mse 仅在 HV 回归时用，此处不用
    head_cfg = {
        "Nuclei-INST": {"type": args.kd_type, "T": args.temperature, "weight": 1.0},
        "Nuclei-TYPE": {"type": args.kd_type, "T": args.temperature, "weight": 1.0},
        "Gland-INST":  {"type": args.kd_type, "T": args.temperature, "weight": 1.0},
        "Gland-TYPE":  {"type": args.kd_type, "T": args.temperature, "weight": 1.0},
        "Patch-Class": {"type": args.kd_type, "T": args.temperature, "weight": 0.5},
        "Lumen-INST":  {"type": args.kd_type, "T": args.temperature, "weight": 0.3, "pseudo": True, "enabled": args.enable_lumen_pseudo},
    }
    # 管腔辅头在 teacher/student 可能不存在时自动跳过
    criterion = DistillLoss(
        gt_weight=0.0 if args.no_gt else args.gt_weight,
        output_weight=args.kd_weight,
        patch_gt_weight=args.patch_gt_weight,
        output_head_cfg=head_cfg,
    ).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # --- data: 真实3源混合，缺的自动回落 Dummy ---
    try:
        from distill.dataset import UnifiedDistillDataset, collate_mixed
        has_real = any(os.path.isdir(p) for p in [args.lizard_root, args.glas_root, args.kather_root])
        if has_real:
            dataset = UnifiedDistillDataset(
                lizard_root=args.lizard_root, glas_root=args.glas_root,
                kather_root=args.kather_root, img_size=args.img_size)
            if len(dataset)==0:
                print("[warn] 真实数据集长度0，回落 Dummy")
                dataset = DummyGTDataset(length=32, img_size=args.img_size)
                loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
            else:
                loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mixed, num_workers=0)
                print(f"[data] Unified lizard+GlaS+Kather -> len {len(dataset)} seg/patch混合 0.7/0.3，管腔伪 {'开0.3' if args.enable_lumen_pseudo else '关'}")
        else:
            dataset = DummyGTDataset(length=32, img_size=args.img_size)
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
            print("[data] 无真实数据目录，用 DummyGTDataset 演示")
    except Exception as e:
        print(f"[data] 加载真实数据失败 {e}，回落 Dummy")
        dataset = DummyGTDataset(length=32, img_size=args.img_size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # --- train loop (最小演示) ---
    # 文件日志：output_dir/train.log + train.csv
    import csv
    log_path = os.path.join(args.output_dir, "train.log")
    csv_path = os.path.join(args.output_dir, "train.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = None
    student.train()
    for epoch in range(args.epochs):
        for step, (imgs, gt_dict) in enumerate(loader):
            imgs = imgs.to(device)
            # gt_dict 内含 _has (dict)，需递归 .to
            def _to_device(d):
                out={}
                for k,v in d.items():
                    if k=="_has":
                        out[k]={kk: vv.to(device) for kk,vv in v.items()}
                    else:
                        out[k]=v.to(device)
                return out
            gt_dict = _to_device(gt_dict)

            # Teacher 前向 (无梯度)
            with torch.no_grad():
                teacher_out = teacher(imgs, train_decoder_list=train_decoder_list)

            # Student 前向
            student_out = student(imgs, train_decoder_list=train_decoder_list)

            # 损失：GT + Output KD
            loss, log = criterion(student_out, teacher_out, gt_dict if not args.no_gt else None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 2 == 0:
                msg = f"epoch {epoch+1} step {step} | L_total {log['L_total'].item():.4f} | L_gt {log['L_gt'].item():.4f} | L_output {log['L_output'].item():.4f}"
                # 细分
                for k in sorted(log.keys()):
                    if k.startswith("GT/") or k.startswith("KD/"):
                        msg += f" | {k} {log[k].item():.3f}"
                print(msg)
                with open(log_path, "a") as lf:
                    lf.write(msg+"\n")
            # 写 csv（每步）
            row = {"epoch": epoch+1, "step": step, "L_total": log["L_total"].item(), "L_gt": log["L_gt"].item(), "L_output": log["L_output"].item()}
            for k in sorted(log.keys()):
                if k.startswith("GT/") or k.startswith("KD/"):
                    row[k]=log[k].item()
            if csv_w is None:
                csv_w = csv.DictWriter(csv_f, fieldnames=list(row.keys()))
                csv_w.writeheader()
            # 若后续 step 新增 key，忽略
            csv_w.writerow({k: row.get(k,"") for k in csv_w.fieldnames})
            csv_f.flush()

        # 每 epoch 存一次
        sd = {"module." + k: v for k, v in student.state_dict().items()}
        torch.save({"desc": sd}, os.path.join(args.output_dir, "weights.tar"))
        # settings.yml 供 run_infer_tile.py 使用
        new_paramset = {k: v for k, v in run_paramset.items() if k != "model_kwargs"}
        new_model_args = dict(decoder_kwargs=decoder_kwargs,
                              considered_tasks=considered_tasks,
                              encoder_backbone_name=args.backbone,
                              backbone_imagenet_pretrained=not args.no_pretrained,
                              fullnet_custom_pretrained=False)
        new_paramset["model_kwargs"] = new_model_args
        with open(os.path.join(args.output_dir, "settings.yml"), "w") as f:
            yaml.safe_dump(new_paramset, f, sort_keys=False)
        print(f"saved -> {args.output_dir}  (epoch {epoch+1})")
        with open(log_path, "a") as lf:
            lf.write(f"saved -> {args.output_dir}  (epoch {epoch+1})\n")

    csv_f.close()
    print(f"Done. Best pattern: L_total = {args.gt_weight}*L_gt + {args.kd_weight}*L_output({args.kd_type} T={args.temperature})")
    print(f"  GT heads  : Nuclei-INST/TYPE, Gland-INST/TYPE, Patch-Class (w={args.patch_gt_weight})")
    print(f"  KD heads  : Nuclei-INST(mse T1) + Nuclei-TYPE/Gland/Patch({args.kd_type} T{args.temperature}) + Lumen伪={args.enable_lumen_pseudo}")
    print("后续扩展: 在 distill/losses.py 的 DistillLoss 里打开 feature/task/relation 插槽即可，不用改本脚本循环")


if __name__ == "__main__":
    main()
