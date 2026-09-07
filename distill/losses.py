"""
五类归属蒸馏损失 — 模板 v0：只实现 GT + Output KD，预留扩展口

按 /桌面/Cerberus蒸馏方案_有GT无GT分流_五类归属.md  §1 五类总览：

  GT         真标签纠偏         (对应 A0)
  Output KD  输出分布           KL / CS  T=4
  Feature KD 预留 Hint/Review/FKD/IFVD
  Task KD    预留 OTW/AFD
  Relation   预留 PA/CIRKD/CC/HO

当前 v0 只开启:
  L_total = L_gt (4头) + λ_output * L_output_kd (KL或CS, T可配)

GT 定义 (D_l 真GT集, 管腔不算GT):
  L_gt_nuc_seg = Dice + CE + MSE(HV)  — 这里简化成 Dice+CE，三通道(IP-ERODED-CONTOUR-3)
  L_gt_nuc_cls = CE + Dice per-class
  L_gt_gland   = Dice + CE
  L_gt_patch   = CE (Kather 9类)
  L_gt = L_nuc_seg + L_nuc_cls + L_gland + 0.5*L_patch   (Patch 0.5防喧宾夺主)

Output KD (5头训练期都可算, v0先做4头真GT头):
  核分类/腺体/Patch : CS散度 KDAM T=4  weight 1.0/1.0/0.5
  核二值/HV         : Sigmoid MSE       T=1
  管腔辅头伪        : CS伪 T=4 weight 0.3 (v0暂不启用，预留开关)

所有损失都支持按像素/样本的 has_flag 跳过 (无GT的图对应头不计loss)。

扩展预留:
  往 LOSS_REGISTRY 加新类，然后在 DistillLossConfig 里打开开关即可，
  不用改训练循环。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def dice_loss_per_head(pred, target_onehot, smooth=1e-3):
    """pred: (B,C,H,W) logits, target_onehot: (B,C,H,W) 0/1 float.
    适配 models/utils/loss_utils 的 dice 但支持任意 C。"""
    prob = F.softmax(pred, dim=1)
    # 与 loss_utils 一致：按通道求和再 sum
    inse = torch.sum(prob * target_onehot, dim=(0, 2, 3))
    l = torch.sum(prob, dim=(0, 2, 3))
    r = torch.sum(target_onehot, dim=(0, 2, 3))
    dice = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    return dice.sum()  # 标量


def ce_loss(pred, target, weight=None):
    """pred (B,C,H,W) logits, target (B,H,W) long."""
    # target 可能带 channel 维 (B,1,H,W) -> squeeze
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    target = target.long()
    return F.cross_entropy(pred, target, weight=weight)


# ---------------------------------------------------------------------------
# GT losses  — 归属 GT (§3.1)
# ---------------------------------------------------------------------------
class GTLoss(nn.Module):
    """4头GT汇总。可通过 config 单独开关每头、调 Patch 权重。"""
    def __init__(self, patch_weight=0.5, class_weights=None):
        super().__init__()
        self.patch_weight = patch_weight
        self.class_weights = class_weights  # dict head -> Tensor

    def forward(self, student_out, gt_dict, has_flag=None):
        """
        student_out: dict  Cerberus NetDesc forward 输出，key 如 "Nuclei-INST","Nuclei-TYPE","Gland-INST","Patch-Class"
        gt_dict: dict  同key对应GT，value为 Tensor
            - 分割头: (B,C,H,W) logits 对应的 target
              * 对于 CE: (B,H,W) long，存为 gt_dict[head]  (已argmax)
              * 对于 Dice: (B,C,H,W) onehot float，存为 gt_dict[head+"_onehot"] 可选
            实际调用时见下面的 build_gt_batch 注释。
        has_flag: dict head -> BoolTensor (B,) 是否有GT，没有则跳过

        为简化模板，这里约定:
          gt_dict = {
            "Nuclei-INST": (B,H,W) long  (0/1/2 三类: bg/inner/contour),
            "Nuclei-TYPE": (B,H,W) long  (0..6 含bg),
            "Gland-INST":  (B,H,W) long  (0/1/2),
            "Gland-TYPE":  (B,H,W) long  (0..2),
            "Patch-Class": (B,)    long  (0..8)
          }
        若某头缺失则自动跳过。
        """
        total = 0.0
        details = {}

        # Nuclei-INST : 3类分割 CE + Dice
        if "Nuclei-INST" in student_out and "Nuclei-INST" in gt_dict:
            pred = student_out["Nuclei-INST"]  # (B,3,H,W)
            tgt = gt_dict["Nuclei-INST"]       # (P,H,W) P<=B
            # 若 batch 混合，部分样本无此头，则按 mask 切 pred
            if has_flag is not None and "Nuclei-INST" in has_flag:
                pred = pred[has_flag["Nuclei-INST"]]
            if has_flag is None or has_flag.get("Nuclei-INST", torch.ones(1, device=student_out["Nuclei-INST"].device, dtype=torch.bool)).any():
                loss_ce = ce_loss(pred, tgt, weight=self.class_weights.get("Nuclei-INST") if self.class_weights else None)
                # dice 需要 onehot
                with torch.no_grad():
                    tgt_onehot = F.one_hot(tgt.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
                loss_dice = dice_loss_per_head(pred, tgt_onehot)
                loss = loss_ce + loss_dice
                details["Nuclei-INST"] = loss.detach()
                total = total + loss

        # Nuclei-TYPE : 7类分割
        if "Nuclei-TYPE" in student_out and "Nuclei-TYPE" in gt_dict:
            pred = student_out["Nuclei-TYPE"]
            tgt = gt_dict["Nuclei-TYPE"]
            if has_flag is not None and "Nuclei-TYPE" in has_flag:
                pred = pred[has_flag["Nuclei-TYPE"]]
            loss_ce = ce_loss(pred, tgt, weight=self.class_weights.get("Nuclei-TYPE") if self.class_weights else None)
            with torch.no_grad():
                tgt_onehot = F.one_hot(tgt.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
            loss_dice = dice_loss_per_head(pred, tgt_onehot)
            loss = loss_ce + loss_dice
            details["Nuclei-TYPE"] = loss.detach()
            total = total + loss

        # Gland-INST : 3类
        if "Gland-INST" in student_out and "Gland-INST" in gt_dict:
            pred = student_out["Gland-INST"]
            tgt = gt_dict["Gland-INST"]
            if has_flag is not None and "Gland-INST" in has_flag:
                pred = pred[has_flag["Gland-INST"]]
            loss_ce = ce_loss(pred, tgt)
            with torch.no_grad():
                tgt_onehot = F.one_hot(tgt.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
            loss_dice = dice_loss_per_head(pred, tgt_onehot)
            loss = loss_ce + loss_dice
            details["Gland-INST"] = loss.detach()
            total = total + loss

        # Gland-TYPE : 3类
        if "Gland-TYPE" in student_out and "Gland-TYPE" in gt_dict:
            pred = student_out["Gland-TYPE"]
            tgt = gt_dict["Gland-TYPE"]
            if has_flag is not None and "Gland-TYPE" in has_flag:
                pred = pred[has_flag["Gland-TYPE"]]
            loss_ce = ce_loss(pred, tgt)
            with torch.no_grad():
                tgt_onehot = F.one_hot(tgt.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
            loss_dice = dice_loss_per_head(pred, tgt_onehot)
            loss = loss_ce + loss_dice
            details["Gland-TYPE"] = loss.detach()
            total = total + loss

        # Patch-Class : (B,9)  全局分类，无空间维
        if "Patch-Class" in student_out and "Patch-Class" in gt_dict:
            pred = student_out["Patch-Class"]  # (B,9,1,1) or (B,9)
            if pred.dim() == 4:
                pred = pred.squeeze(-1).squeeze(-1)  # -> (B,9)
            if has_flag is not None and "Patch-Class" in has_flag:
                pred = pred[has_flag["Patch-Class"]]
            tgt = gt_dict["Patch-Class"]  # (P,)
            loss = F.cross_entropy(pred, tgt.long())
            details["Patch-Class"] = loss.detach()
            total = total + self.patch_weight * loss

        # Lumen-INST 按B方案：训练期辅头不计GT，这里不算
        return total, details


# ---------------------------------------------------------------------------
# Output KD  — 归属 Output KD (§3.2)
#  支持 KL(T) 和 CS(T) 两种，CS 比 KL 对噪声更稳
# ---------------------------------------------------------------------------
def kd_kl_loss(student_logits, teacher_logits, T=4.0):
    """Standard KL divergence KD. 按像素/样本平均，*T^2."""
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction='mean') * (T * T)


def kd_cs_loss(student_logits, teacher_logits, T=4.0, eps=1e-8):
    """Cauchy-Schwarz divergence (KDAM):
    D_cs = -log( sum(Pt*Ps) / sqrt(sum(Pt^2)*sum(Ps^2)) )
    Pt = softmax(teacher/T), Ps = softmax(student/T)
    对分类/分割都按 (B,C,H,W) 或 (B,C) 逐位置计算后平均.
    数值稳定：分母加 eps，log 内 clamp.
    """
    pt = F.softmax(teacher_logits / T, dim=1)
    ps = F.softmax(student_logits / T, dim=1)
    # sum over channel dim = 1, keep spatial
    # pt,ps: (B,C,H,W) or (B,C)
    # 对于 4D: sum over C -> (B,H,W), 再平均
    # 对于 2D: sum over C -> (B,)
    dot = torch.sum(pt * ps, dim=1)  # (B,H,W) or (B,)
    pt2 = torch.sum(pt * pt, dim=1)
    ps2 = torch.sum(ps * ps, dim=1)
    denom = torch.sqrt(pt2 * ps2 + eps) + eps
    cos = dot / denom  # (B,H,W)
    cos = torch.clamp(cos, min=1e-6, max=1.0)
    loss = -torch.log(cos)
    return loss.mean()


def kd_mse_loss(student_logits, teacher_logits):
    """回归式 KD：Sigmoid MSE，用于 HV / 二值分割 T=1 的场景。
    这里简化：直接对 logits 做 MSE (或 prob MSE 亦可，选 logits 更稳)。"""
    return F.mse_loss(student_logits, teacher_logits)


OUTPUT_KD_FN = {
    'kl': kd_kl_loss,
    'cs': kd_cs_loss,
    'mse': kd_mse_loss,
}


class OutputKDLoss(nn.Module):
    """Output KD 汇总，支持 KL/CS/MSE 按头选择，支持管腔伪权重 0.3。

    config 预期:
      head_cfg = {
        "Nuclei-TYPE": {"type": "cs", "T": 4, "weight": 1.0},
        "Gland-INST":  {"type": "cs", "T": 4, "weight": 1.0},
        "Patch-Class": {"type": "cs", "T": 4, "weight": 0.5},
        "Lumen-INST":  {"type": "cs", "T": 4, "weight": 0.3, "pseudo": True},
        ...
      }
      默认：分类头 cs T4，分割二值头若配 mse 则用 mse，否则 cs。
    """
    def __init__(self, head_cfg=None, default_type='cs', default_T=4.0):
        super().__init__()
        if head_cfg is None:
            # v0 默认：4真头 cs，Patch 0.5；Nuclei-INST也用cs（比mse稳，logits尺度不敏感）
            head_cfg = {
                "Nuclei-INST": {"type": "cs", "T": 4, "weight": 1.0},
                "Nuclei-TYPE": {"type": "cs", "T": 4, "weight": 1.0},
                "Gland-INST":  {"type": "cs", "T": 4, "weight": 1.0},
                "Gland-TYPE":  {"type": "cs", "T": 4, "weight": 1.0},
                "Patch-Class": {"type": "cs", "T": 4, "weight": 0.5},
                # 管腔辅头伪蒸馏，v0默认关闭，需要时把 enabled 打开
                "Lumen-INST":  {"type": "cs", "T": 4, "weight": 0.3, "pseudo": True, "enabled": False},
            }
        self.head_cfg = head_cfg
        self.default_type = default_type
        self.default_T = default_T

    def forward(self, student_out, teacher_out, has_flag=None):
        total = 0.0
        details = {}
        for head, cfg in self.head_cfg.items():
            if not cfg.get("enabled", True):
                continue
            if head not in student_out or head not in teacher_out:
                continue
            # D_u 的管腔伪：置信过滤可在外层做，这里只按权重算
            s = student_out[head]
            t = teacher_out[head]
            ktype = cfg.get("type", self.default_type)
            T = cfg.get("T", self.default_T)
            w = cfg.get("weight", 1.0)
            fn = OUTPUT_KD_FN[ktype]
            if ktype == 'mse':
                loss = fn(s, t)
            else:
                loss = fn(s, t, T)
            details[head] = loss.detach()
            total = total + w * loss
        return total, details


# ---------------------------------------------------------------------------
# 总蒸馏损失 — 模板 v0  (GT + Output KD)
#  预留 Feature/Task/Relation 插槽
# ---------------------------------------------------------------------------
class DistillLoss(nn.Module):
    """模板 v0： L_total = w_gt * L_gt + w_output * L_output

    后续扩展：把 feature_loss / task_loss / relation_loss 做成可选 nn.Module，
    在 forward 里 if enabled 再累加，保持训练循环不变。

    使用:
      crit = DistillLoss(gt_weight=1.0, output_weight=10.0, output_head_cfg={...})
      loss, log = crit(student_out, teacher_out, gt_dict)
      loss.backward()

    约定:
      - student_out / teacher_out: Cerberus NetDesc forward 的 OrderedDict
      - gt_dict: 见 GTLoss 说明
      - teacher_out 已 detach，外面无需再 no_grad (但建议外层 teacher.eval+no_grad)
    """
    def __init__(self,
                 gt_weight=1.0,
                 output_weight=10.0,
                 patch_gt_weight=0.5,
                 output_head_cfg=None,
                 # 预留
                 feature_weight=0.0,
                 task_weight=0.0,
                 relation_weight=0.0):
        super().__init__()
        self.gt_weight = gt_weight
        self.output_weight = output_weight
        self.feature_weight = feature_weight
        self.task_weight = task_weight
        self.relation_weight = relation_weight

        self.gt_loss = GTLoss(patch_weight=patch_gt_weight)
        self.output_loss = OutputKDLoss(head_cfg=output_head_cfg)

        # 预留：后续 Feature/Task/Relation 模块放这里
        self.feature_loss = None
        self.task_loss = None
        self.relation_loss = None

    def forward(self, student_out, teacher_out, gt_dict=None):
        log = {}
        total = 0.0

        # GT
        if gt_dict is not None and self.gt_weight != 0:
            has_flag = gt_dict.get("_has", None)
            # 去掉 _has 再传给 GTLoss
            gt_for_loss = {k:v for k,v in gt_dict.items() if k!="_has"}
            l_gt, d_gt = self.gt_loss(student_out, gt_for_loss, has_flag=has_flag)
            log["L_gt"] = l_gt.detach()
            for k, v in d_gt.items():
                log[f"GT/{k}"] = v
            total = total + self.gt_weight * l_gt
        else:
            log["L_gt"] = torch.tensor(0.0, device=next(iter(student_out.values())).device)

        # Output KD
        if self.output_weight != 0:
            l_out, d_out = self.output_loss(student_out, teacher_out)
            log["L_output"] = l_out.detach()
            for k, v in d_out.items():
                log[f"KD/{k}"] = v
            total = total + self.output_weight * l_out
        else:
            log["L_output"] = torch.tensor(0.0, device=next(iter(student_out.values())).device)

        # 预留 Feature / Task / Relation (当前为 0)
        # if self.feature_loss is not None:
        #     l_f, d_f = self.feature_loss(student_feats, teacher_feats, gt_dict)
        #     total += self.feature_weight * l_f

        log["L_total"] = total.detach()
        return total, log
