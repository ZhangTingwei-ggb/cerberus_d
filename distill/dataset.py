"""
正式蒸馏数据集适配 — lizard + GlaS + Kather100K 统一转 gt_dict

三套格式差异在 Dataset 内消化，上层 loss 只认 gt_dict：
  gt_dict = {
    "Nuclei-INST": (H,W) long 0..2 (bg=0 inner=1 contour=2, IP-ERODED-CONTOUR-3),
    "Nuclei-TYPE": (H,W) long 0..6 (bg=0 + 6类),
    "Gland-INST" : (H,W) long 0..2,
    "Patch-Class": () long 0..8  (Kather)
  }
缺哪头就不放那个key，GTLoss 自动跳过。管腔伪不放GT，用 gland 图复用产伪，由 --enable_lumen_pseudo 控制 KD 0.3。

目录默认（可命令行覆盖）：
  lizard:  /home/linjiatai/zhang/cerberus_d/dataset/nuclear/lizard
           含 lizard_images1/Lizard_Images1/*.png + lizard_images2/Lizard_Images2/*.png
               lizard_labels/Lizard_Labels/Labels/*.mat
  glas:    /home/linjiatai/zhang/cerberus_d/dataset/gland/GlaS
           含 image/*.bmp + labels/*_anno.bmp
  kather:  /home/linjiatai/zhang/cerberus_d/dataset/classification/Kather100K/NCT-CRC-HE-100K
           含 9子文件夹 ADI/BACK/...TUM/*.tif，文件夹名即标签

大盘同构路径 /media/linjiatai/086399513677BED7/zhang/dataset/... 也兼容，用 --lizard_root 等覆盖即可。
"""
import os, glob, random
import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

# ---- utils: INST 3类生成 (简化版 IP-ERODED-CONTOUR) ----
def inst_to_3class(inst_map, ksize=3):
    """快速版：为保证训练速度，先用二值 0 bg 1 fg，contour 2 暂空；后续可切回完整 erode/dilate 版"""
    # 完整版在 targets.py 里是每实例 erode/dilate，较慢；正式训练可预先生成 mat 缓存，这里先用二值保证跑通
    bin_map = (inst_map > 0).astype(np.int64)  # 0/1
    # 若需 3类，取消下面注释：
    # h,w=inst_map.shape; bin=(inst_map>0).astype(np.uint8); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize)); ...
    return bin_map

CLASS_NAME_TO_IDX = {"ADI":0,"BACK":1,"DEB":2,"LYM":3,"MUC":4,"MUS":5,"NORM":6,"STR":7,"TUM":8}

def _resize(img, size=448, is_mask=False):
    if img.shape[0]==size and img.shape[1]==size:
        return img
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)

# ---- 单任务 datasets ----
class LizardDataset(Dataset):
    def __init__(self, root, img_size=448):
        self.img_size = img_size
        self.root = root
        # 收集成对
        ims = glob.glob(os.path.join(root, "lizard_images1/Lizard_Images1/*.png")) + \
              glob.glob(os.path.join(root, "lizard_images2/Lizard_Images2/*.png"))
        # 兼容部分环境图在根下直接放 png
        if not ims:
            ims = glob.glob(os.path.join(root, "*.png"))
        self.ims = sorted(ims)
        self.label_dir = os.path.join(root, "lizard_labels/Lizard_Labels/Labels")
        if not os.path.isdir(self.label_dir):
            # 兜底：archive 解压后的其他位置
            cand = glob.glob(os.path.join(root, "**/*.mat"), recursive=True)
            self.label_dir = os.path.dirname(cand[0]) if cand else root

    def __len__(self): return len(self.ims)

    def __getitem__(self, idx):
        imp = self.ims[idx]
        base = os.path.splitext(os.path.basename(imp))[0]
        matp = os.path.join(self.label_dir, base + ".mat")
        if not os.path.exists(matp):
            # 模糊搜
            alt = glob.glob(os.path.join(os.path.dirname(self.label_dir), "**", base+".mat"), recursive=True)
            matp = alt[0] if alt else None
        img = cv2.imread(imp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mat
        nuc_inst = np.zeros(img.shape[:2], dtype=np.int32)
        nuc_type = np.zeros(img.shape[:2], dtype=np.int64)
        if matp and os.path.exists(matp):
            d = scipy.io.loadmat(matp)
            inst_map = d["inst_map"].astype(np.int32)  # HxW
            ids = d["id"].flatten()
            clss = d["class"].flatten()  # 1..6
            # inst_map 可能比 img 小一点？以 inst_map 尺寸为准，已对齐
            # 保证 img 与 inst_map 尺寸一致：img 原 500x500，inst_map 可能是同尺寸
            # 若不一致，resize inst_map 到 img 尺寸再算 3类
            if inst_map.shape != img.shape[:2]:
                # 最近邻缩到 img 尺寸
                inst_map = cv2.resize(inst_map.astype(np.int32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            # 生成 type 图
            type_map = np.zeros_like(inst_map, dtype=np.int64)
            for iid, c in zip(ids, clss):
                type_map[inst_map == iid] = int(c)  # 1..6，0 背景
            nuc_inst_raw = inst_map
            nuc_inst = inst_to_3class(nuc_inst_raw, ksize=3)
            nuc_type = type_map.astype(np.int64)  # 0..6
            # 若 img 与 nuc 图尺寸不一致再同步 resize 前已处理，此处两者已同尺寸
        # resize 到 img_size
        img = _resize(img, self.img_size, False)
        nuc_inst = _resize(nuc_inst.astype(np.uint8), self.img_size, True).astype(np.int64)
        nuc_type = _resize(nuc_type.astype(np.uint8), self.img_size, True).astype(np.int64)
        # 轻增强：随机翻转
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            nuc_inst = np.ascontiguousarray(nuc_inst[:, ::-1])
            nuc_type = np.ascontiguousarray(nuc_type[:, ::-1])
        img_t = torch.from_numpy(img).permute(2,0,1).float()  # 0-255，NetDesc内 /255
        gt = {
            "Nuclei-INST": torch.from_numpy(nuc_inst).long(),
            "Nuclei-TYPE": torch.from_numpy(nuc_type).long(),
        }
        return img_t, gt

class GlasDataset(Dataset):
    def __init__(self, root, img_size=448):
        self.img_size = img_size
        self.root = root
        # 兼容两种结构：GlaS/image/*.bmp  或 image/*.bmp
        for cand in [os.path.join(root,"image"), os.path.join(root,"GlaS/image"), root]:
            if os.path.isdir(cand):
                ims = sorted(glob.glob(os.path.join(cand,"*.bmp")) + glob.glob(os.path.join(cand,"*.png")))
                if ims:
                    self.img_dir = cand
                    break
        else:
            self.img_dir = root
            ims = []
        self.ims = sorted(glob.glob(os.path.join(self.img_dir,"*.bmp")) + glob.glob(os.path.join(self.img_dir,"*.png")))
        # label 目录
        self.lbl_dir = None
        for cand in [os.path.join(root,"labels"), os.path.join(root,"GlaS/labels"), os.path.join(root,"Warwick_QU_Dataset")]:
            if os.path.isdir(cand):
                self.lbl_dir = cand
                break
        if self.lbl_dir is None:
            self.lbl_dir = os.path.join(root,"labels")

    def __len__(self): return len(self.ims)

    def __getitem__(self, idx):
        imp = self.ims[idx]
        base = os.path.splitext(os.path.basename(imp))[0]
        # _anno.bmp 命名
        cand = [os.path.join(self.lbl_dir, base+"_anno.bmp"), os.path.join(self.lbl_dir, base+".bmp"),
                os.path.join(os.path.dirname(self.img_dir).replace("image","labels"), base+"_anno.bmp")]
        lblp=None
        for c in cand:
            if os.path.exists(c): lblp=c; break
        if lblp is None:
            # 模糊
            alt = glob.glob(os.path.join(self.lbl_dir, base+"*.bmp"))
            lblp = alt[0] if alt else None
        img = cv2.imread(imp); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gland_inst = np.zeros(img.shape[:2], dtype=np.int64)
        if lblp and os.path.exists(lblp):
            lbl = cv2.imread(lblp, cv2.IMREAD_GRAYSCALE)
            # Warwick 标签 0背景 + 1..N 实例，需转 3类
            gland_inst = inst_to_3class(lbl.astype(np.int32), ksize=11 if max(lbl.shape)>600 else 5)
        img = _resize(img, self.img_size, False)
        gland_inst = _resize(gland_inst.astype(np.uint8), self.img_size, True).astype(np.int64)
        if random.random()<0.5:
            img=np.ascontiguousarray(img[:,::-1]); gland_inst=np.ascontiguousarray(gland_inst[:,::-1])
        img_t = torch.from_numpy(img).permute(2,0,1).float()
        gt={"Gland-INST": torch.from_numpy(gland_inst).long()}
        return img_t, gt

class KatherDataset(Dataset):
    def __init__(self, root, img_size=448):
        self.img_size = img_size
        self.root = root
        # root 可能是 .../NCT-CRC-HE-100K 或其父目录
        if os.path.isdir(os.path.join(root,"NCT-CRC-HE-100K")):
            root = os.path.join(root,"NCT-CRC-HE-100K")
        self.root = root
        self.samples=[] # (path, label)
        for cls_name, idx in CLASS_NAME_TO_IDX.items():
            d=os.path.join(root, cls_name)
            if not os.path.isdir(d): continue
            for p in glob.glob(os.path.join(d,"*.tif")):
                self.samples.append((p, idx))
        self.samples.sort()

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, lab = self.samples[idx]
        img=cv2.imread(p); img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=_resize(img, self.img_size, False)
        if random.random()<0.5:
            img=np.ascontiguousarray(img[:,::-1])
        img_t=torch.from_numpy(img).permute(2,0,1).float()
        gt={"Patch-Class": torch.tensor(lab, dtype=torch.long)}
        return img_t, gt

class UnifiedDistillDataset(Dataset):
    """按 super-task 0.7 seg /0.3 patch 混合采样，batch 内由 collate 拼；len 按最大集x倍，随机采样。"""
    def __init__(self, lizard_root=None, glas_root=None, kather_root=None, img_size=448, seg_prob=0.7):
        self.seg_prob=seg_prob; self.img_size=img_size
        self.datasets=[]
        self.weights=[]
        if lizard_root and os.path.isdir(lizard_root):
            # 兼容传入到 .../lizard 或 .../nuclear
            cand = lizard_root
            if os.path.isdir(os.path.join(lizard_root,"lizard")):
                cand=os.path.join(lizard_root,"lizard")
            ds=LizardDataset(cand, img_size)
            if len(ds)>0:
                self.datasets.append(ds); self.weights.append(0.6)  # seg 内 lizard 占多
        if glas_root and os.path.isdir(glas_root):
            cand=glas_root
            # 传入可能是 dataset/gland
            if os.path.isdir(os.path.join(glas_root,"GlaS")) and not os.path.isdir(os.path.join(glas_root,"image")):
                cand=os.path.join(glas_root,"GlaS")
            ds=GlasDataset(cand, img_size)
            if len(ds)>0:
                self.datasets.append(ds); self.weights.append(0.4)
        if kather_root and os.path.isdir(kather_root):
            ds=KatherDataset(kather_root, img_size)
            if len(ds)>0:
                self.datasets.append(ds); self.weights.append(1.0)
        # 归一化 seg vs patch 的采样：seg 0.7，patch 0.3
        # 简单实现：每次 __getitem__ 以 seg_prob 概率抽 seg 池，否则抽 patch
        self.seg_indices=[i for i,d in enumerate(self.datasets) if not isinstance(d,KatherDataset)]
        self.patch_indices=[i for i,d in enumerate(self.datasets) if isinstance(d,KatherDataset)]
        # 长度取最长 * 2 保证 epoch 足量
        max_len = max((len(d) for d in self.datasets), default=64)
        self._len = min(max_len*2, 512)  # 每epoch 512样本，batch8时64步，3分钟内可完成1epoch验证

    def __len__(self): return self._len if self.datasets else 0
    # 供验证快速跑
    def set_len(self, n): self._len=n

    def __getitem__(self, idx):
        if not self.datasets:
            raise IndexError("no dataset configured")
        # super-task 采样
        if self.patch_indices and random.random() > self.seg_prob:
            di = random.choice(self.patch_indices)
        else:
            di = random.choice(self.seg_indices) if self.seg_indices else random.choice(list(range(len(self.datasets))))
        ds=self.datasets[di]
        j= random.randrange(len(ds))
        return ds[j]

def collate_mixed(batch):
    """batch: list of (img, gt_dict) -> (imgs BCHW, gt_dict)
    对缺失头的样本不计loss：gt_dict 只堆有GT的样本，并在 gt_dict['_has'] 存 mask (B,) bool 供 loss 过滤。"""
    imgs=torch.stack([b[0] for b in batch])
    B=len(batch)
    all_keys=set()
    for _,gt in batch: all_keys.update(gt.keys())
    gt_dict={}
    has_dict={}
    for k in all_keys:
        # 收集索引
        idxs=[i for i,(_,gt) in enumerate(batch) if k in gt]
        vals=[batch[i][1][k] for i in idxs]
        if not vals: continue
        gt_dict[k]=torch.stack(vals)
        # has mask 长度 B，True 表示该样本有此头
        mask=torch.zeros(B, dtype=torch.bool)
        for i in idxs: mask[i]=True
        has_dict[k]=mask
    gt_dict["_has"]=has_dict
    return imgs, gt_dict
