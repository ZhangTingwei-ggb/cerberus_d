# Cerberus 知识蒸馏 — 从无GT到有GT的五类框架（正式训练版）

> Teacher: Cerberus ResNet34 冻结 checkpoint/resnet34_cerberus (27M, 6头)
> Student: resnet18 / enet / mobilenet_v3_large 三选一，共享 Cerberus 5尺度解码头
> 方案对齐: 桌面 Cerberus蒸馏方案_有GT无GT分流_五类归属.md + xlsx

---

## 0. 你手上有什么

Cerberus = 1个 backbone + 6个解码头，多任务，输入 448x448，输出：

  Gland-INST  3ch 腺体实例  | Gland-TYPE  3ch 腺体类型
  Lumen-INST  3ch 管腔实例  | Nuclei-INST 3ch 核实例
  Nuclei-TYPE 7ch 核类型    | Patch-Class 9ch 组织分类(ADI/BACK/DEB/LYM/MUC/MUS/NORM/STR/TUM)

Teacher 权重: checkpoint/resnet34_cerberus/weights.tar + settings.yml

Student 三选一（已适配 5尺度 [stride1,2,4,8,16]）：

  resnet18           18.3M  filters [64,64,128,256,512]   ImageNet预训练  batch建议4
  enet               1.92M  filters [16,16,64,128,128]   随机初始化      batch建议8
  mobilenet_v3_large 3.48M  filters [16,16,24,40,960]    ImageNet预训练  batch建议8
  mobilenet_v3_small 1.2M   filters [16,16,16,24,576]    旧soft-label脚本仍支持

---

## 1. 五类归属总览（对应方案 §1）

  类别       作用               本仓库落点
  GT         真标签纠Teacher噪声   核2头+腺体+Patch(Kather100K) 4头真GT，管腔无GT
  Output KD  学Teacher输出分布    4真头 CS(T=4) + 管腔辅头伪CS 0.3
  Feature KD 学中间特征            预留 Hint/Review/FKD/IFVD (losses.py 插槽)
  Task KD    多任务调度            预留 OTW/AFD
  Relation   像素/跨图关系          预留 PA/CIRKD/CC/HO

当前 GT + Output KD，对应消融 A2：

  L_total = L_gt(核+腺体+Patch*0.5) + 10 * L_output(CS或KL, T=4)
  L_gt = L_nuc_seg(Dice+CE) + L_nuc_cls(Dice+CE) + L_gland(Dice+CE) + 0.5*L_patch(CE)
  L_output = Σ KD(Gland/Nuclei/Patch, CS T4) + 0.3*KD(Lumen伪, CS T4, 仅--enable_lumen_pseudo)
  管腔辅头：训练期轻量辅头权重0.3，推理期剪掉，不参与GT

---

## 2. 数据组织（真实路径）

大盘 /media/.../dataset 与本地 cerberus_d/dataset 同构，已自动兼容：

  nuclear/lizard/
    lizard_images1/Lizard_Images1/*.png 80张 + lizard_images2/Lizard_Images2/*.png 158张 =238张
    lizard_labels/Lizard_Labels/Labels/*.mat 238个 (inst_map + id/class)
    -> 产 Nuclei-INST(3类, 快速版二值0/1, 可切回完整erode) + Nuclei-TYPE(0..6)

  gland/GlaS/
    image/*.bmp 165张 + labels/*_anno.bmp 165张
    -> 产 Gland-INST 0..2

  classification/Kather100K/NCT-CRC-HE-100K/
    9子文件夹 ADI/BACK/DEB/LYM/MUC/MUS/NORM/STR/TUM/*.tif 各约1万，224->448
    -> 产 Patch-Class 0..8，文件夹名即标签

  lumen/ 空，B方案复用 GlaS 图产管腔伪，不单独读GT

  nuclear/PanNuke Fold1-3 已并入 lizard 的 pannuke 子集，不单独用

数据在 distill/dataset.py 内统一转 gt_dict：

  gt_dict = {
    "Nuclei-INST": (H,W) long 0..2,
    "Nuclei-TYPE": (H,W) long 0..6,
    "Gland-INST":  (H,W) long 0..2,
    "Patch-Class": () long 0..8,
  }
缺哪头就不放该key，GTLoss 按 _has 掩码自动切片，不会计假GT。Patch与分割混合采样 super-task 0.7 seg /0.3 patch，每batch随机，loss已处理不一致batch。

---

## 3. 文件清单与用途

  distill/dataset.py                 正式数据集适配，lizard+GlaS+Kather统一转gt_dict，collate_mixed按掩码堆叠
  distill/losses.py                  五类损失工厂：GTLoss + OutputKDLoss(CS/KL/MSE) + DistillLoss(GT+Output, 预留feature/task/relation插槽)
  distill/distill_train_gt_output.py 正式训练脚本，在线Teacher前向，GT+Output KD，三backbone + KL/CS + 管腔0.3 + 目录参数 + 日志，已验证真实数据1 epoch可存weights
  distill/gen_soft_labels.py         旧流程1：无GT离线产软标签 tile/wsi -> soft_labels/*.pt（有GT时不用，管腔伪离线对照才用）
  distill/distill_train.py           旧流程2：无GT读pt仅Output KD（有GT的A2不用，作为A1基线保留）
  models/backbone/enet.py            新增ENet，5尺度输出
  models/backbone/__init__.py        注册 enet，filter_info [16,16,64,128,128]
  models/backbone/mobilenet.py       已有 mbv3_large/small 5尺度适配
  models/net_desc.py                 修复 Patch-Class 头写死512 -> 自动取filters[-1]

现在正式训练只用：distill_train_gt_output.py + dataset.py + losses.py
旧两个py保留作无GT基线和备用，不删。

---

## 4. 蒸馏命令

在 cerberus_d 根目录，conda linjiatai_4090，单卡：

### 4.1 单次

  python distill/distill_train_gt_output.py \
    --teacher_dir checkpoint/resnet34_cerberus \
    --backbone enet --kd_type cs --temperature 4 --enable_lumen_pseudo \
    --epochs 50 --batch_size 8 --lr 1e-3 \
    --lizard_root dataset/nuclear/lizard --glas_root dataset/gland/GlaS --kather_root dataset/classification/Kather100K \
    --output_dir checkpoint/distill/test1/enet_gt_cs_lumen03 --gpu 0

  python distill/distill_train_gt_output.py \
    --teacher_dir checkpoint/resnet34_cerberus \
    --backbone resnet18 --kd_type cs --temperature 4 --enable_lumen_pseudo \
    --epochs 50 --batch_size 4 --lr 1e-3 \
    --lizard_root dataset/nuclear/lizard --glas_root dataset/gland/GlaS --kather_root dataset/classification/Kather100K \
    --output_dir checkpoint/distill/test1/resnet18_gt_cs_lumen03 --gpu 0
  
  python distill/distill_train_gt_output.py \
    --teacher_dir checkpoint/resnet34_cerberus \
    --backbone mobilenet_v3_large --kd_type cs --temperature 4 --enable_lumen_pseudo \
    --epochs 50 --batch_size 8 --lr 1e-3 \
    --lizard_root dataset/nuclear/lizard --glas_root dataset/gland/GlaS --kather_root dataset/classification/Kather100K \
    --output_dir checkpoint/distill/test1/mobilenet_v3_large_gt_cs_lumen03 --gpu 0

### 4.2 三学生横向对比

  for bb in resnet18 enet mobilenet_v3_large; do
    python distill/distill_train_gt_output.py \
      --teacher_dir checkpoint/resnet34_cerberus --backbone $bb --kd_type cs --temperature 4 --enable_lumen_pseudo \
      --epochs 50 --batch_size 8 --output_dir checkpoint/distill/${bb}_gt_cs_lumen03 --gpu 0
  done

### 4.3 KL vs CS 对比

  python distill/distill_train_gt_output.py --teacher_dir checkpoint/resnet34_cerberus --backbone resnet18 --kd_type kl --output_dir checkpoint/distill/r18_gt_kl --epochs 50
  python distill/distill_train_gt_output.py --teacher_dir checkpoint/resnet34_cerberus --backbone resnet18 --kd_type cs --output_dir checkpoint/distill/r18_gt_cs --epochs 50

CS 公式: Dcs = -ln( ΣPt*Ps / sqrt(ΣPt²·ΣPs²) ), Pt=softmax(teacher/T), Ps=softmax(student/T)，按像素平均，比KL抗Teacher噪声。

### 4.4 参数说明

  --backbone resnet18/enet/mobilenet_v3_large/mobilenet_v3_small/mobilenet_v2
  --kd_type cs(推荐)/kl/mse
  --temperature 4  
  --gt_weight 1.0 
  --kd_weight 10.0 
  --patch_gt_weight 0.5
  --enable_lumen_pseudo 开管腔辅头伪0.3（默认关，B方案开） 
  --no_gt 纯KD退化A1 --no_pretrained
  --epochs 1演示/50正式(enet无预训练建议50-80，resnet/mbv有预训练30-50) 
  --batch_size 见§6 
  --lr 1e-3 
  --gpu 0
  --lizard_root/
  --glas_root/
  --kather_root 默认指向 dataset/.../，大盘路径用绝对路径覆盖，不传回落Dummy

输出：output_dir/weights.tar + settings.yml（可直接给 run_infer_tile.py/wsi.py）+ train.log + train.csv（每步 L_total/L_gt/L_output 及细分）

---

## 5. 日志与输出

  终端每2步打印：epoch X step Y | L_total | L_gt | L_output | GT/... | KD/...
  文件：
    train.log  同终端文本，追加
    train.csv  每步一行 epoch,step,L_total,L_gt,L_output,GT/Gland-INST,... 便于Excel画曲线
    weights.tar 每epoch覆盖保存，Ctrl+C已跑完的epoch不会丢
    settings.yml 同步更新 backbone 名

  查看：
    cat checkpoint/distill/enet_gt_cs_lumen03/train.log
    cat checkpoint/distill/enet_gt_cs_lumen03/train.csv

---

## 6. batch size / epochs 怎么定

  batch：看显存，4090 24G enet/mbv3 batch8稳 resnet18 batch4稳，小于4 BN不稳，12G卡减半。大batch throughput高，小batch用梯度累积等效。
  epochs：enet 50-80，resnet/mbv 30-50，先跑50看train.csv的L_total是否还在降，平台期就停，已存权重即最好。当前每epoch len=512(batch8=64步)，想覆盖更多数据改 distill/dataset.py 的 min(max_len*2, 512) 到 2000（250步/epoch，约2小时/50epoch）。

---

## 7. 旧流程（无GT，仍可用）

  # 1 教师产软标签
  python distill/gen_soft_labels.py --teacher_dir checkpoint/resnet34_cerberus --input_dir /my_histo/tiles --input_type tile --output_dir soft_labels
  # wsi: --input_type wsi --nr_tiles_per_wsi 200
  # 2 学生读pt训练（仅Output KD）
  python distill/distill_train.py --teacher_dir checkpoint/resnet34_cerberus --soft_dir soft_labels --output_dir mobilenet_v3_large_cerberus --backbone mobilenet_v3_large --epochs 20

有GT时不用产pt，在线蒸馏更快且aug一致；无GT或想缓存D_u管腔伪才用旧流程作A1基线。

---

## 8. 推理

  python run_infer_tile.py --gpu=0 --model checkpoint/distill/enet_gt_cs_lumen03 --input_dir dataset/input_tile_d --output_dir dataset/output_tile_d/enet_test --cache_path /tmp/cache --batch_size 16
  python run_infer_wsi.py  --gpu=0 --model checkpoint/distill/enet_gt_cs_lumen03 --input_dir dataset/input_wsi_d --output_dir dataset/output_wsi/enet_test --cache_path /tmp/cache

输出与Teacher一致，可直接对比 overlay/.mat

---

## 9. 损失扩展（后续 Feature/Task/Relation）

  distill/losses.py DistillLoss 已预留：feature_loss/task_loss/relation_loss = None，后续填 Hint/Review/FKD/IFVD/OTW/AFD/PA/CIRKD/CC/HO，forward里 if enabled 累加，不改训练循环。

## 10. 常见问题

  Q: 为何lizard就够，PanNuke Fold还要吗？ A: 论文核只用lizard(含PanNuke/CoNSeP等6源)，lizard 238张即全量，Fold1-3是PanNuke拆散的重复，不用。
  Q: Kather文件夹名即标签？ A: 是，9类映射 0..8 在 dataset.py 中。
  Q: 管腔何时开？ A: B方案训练期开--enable_lumen_pseudo，推理剪头。
  Q: 3类INST现在是二值？ A: 为速度暂用二值0/1，contour 2空，后续切回完整erode/dilate可提边界精度。
