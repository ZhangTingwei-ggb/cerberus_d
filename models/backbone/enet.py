"""ENet backbone adapted for Cerberus 5-scale decoder.

Reference: Paszke et al. "ENet: A Deep Neural Network Architecture for
Real-Time Semantic Segmentation" (arXiv 1606.02147).

Cerberus decoder expects 5 feature maps at strides [1,2,4,8,16]:
  [x0(stride1), x1(stride2), x2(stride4), x3(stride8), x4(stride16)]

ENet natively downsamples 2x -> 4x -> 8x.  We:
  * take x1 from the initial block (stride 2, 16 ch)
  * take x2 after stage1 (stride 4, 64 ch)
  * take x3 after stage2 (stride 8, 128 ch)
  * synthesize x4 by 2x strided conv on x3 -> stride 16 (128 ch)
  * synthesize x0 by 2x bilinear upsample of x1 -> stride 1 (16 ch)

filter_info returned: [16, 16, 64, 128, 128]  (x0..x4 channels)
Compatible with models/utils/net_layers.get_decoder / get_classification_head.

Usage:
  from models.backbone.enet import enet
  backbone = enet(pretrained=False)
  feats = backbone(torch.randn(2,3,448,448))  # list of 5 tensors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class InitialBlock(nn.Module):
    """ENet initial block: 3x3 conv stride2 (13 ch) + 3x3 maxpool, concat -> 16 ch, stride 2."""
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        # ENet initial: conv branch 13 channels, pool branch 3 channels -> concat == 16
        assert out_channels == 16, "ENet initial out_channels must be 16"
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(x)
        out = torch.cat([conv_out, pool_out], dim=1)
        out = self.bn(out)
        out = self.prelu(out)
        return out


class RegularBottleneck(nn.Module):
    """Regular bottleneck (no downsampling)."""
    def __init__(self, channels, internal_ratio=4, dropout_prob=0.1,
                 dilation=1, asymmetric=False, kernel_size=3):
        super().__init__()
        internal_channels = channels // internal_ratio
        # 1x1 projection
        self.conv1 = nn.Conv2d(channels, internal_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU(internal_channels)

        # 3x3 conv (or asymmetric / dilated)
        if asymmetric:
            # 5x1 + 1x5 decomposition (ENet asymmetric)
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, (5, 1), padding=(2, 0), bias=False),
                nn.Conv2d(internal_channels, internal_channels, (1, 5), padding=(0, 2), bias=False),
            )
        else:
            padding = dilation if kernel_size == 3 else 0
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size,
                                   padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU(internal_channels)

        # 1x1 expansion
        self.conv3 = nn.Conv2d(internal_channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.prelu_out = nn.PReLU(channels)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        out = out + identity
        out = self.prelu_out(out)
        return out


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottleneck (stride 2): conv branch + pool branch, concat."""
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // internal_ratio

        # main branch
        self.conv1 = nn.Conv2d(in_channels, internal_channels, 2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU(internal_channels)

        self.conv2 = nn.Conv2d(internal_channels, internal_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU(internal_channels)

        self.conv3 = nn.Conv2d(internal_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # other branch: maxpool + padding with zeros for channel mismatch
        self.pool = nn.MaxPool2d(2, stride=2)
        # if out_channels > in_channels, the pool branch is padded with zeros after pooling
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.prelu_out = nn.PReLU(out_channels)

    def forward(self, x):
        # main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        # other branch
        other = self.pool(x)
        # pad channels with zeros if needed
        if self.out_channels > self.in_channels:
            pad_ch = self.out_channels - self.in_channels
            zeros = torch.zeros(other.shape[0], pad_ch, other.shape[2], other.shape[3],
                                device=other.device, dtype=other.dtype)
            other = torch.cat([other, zeros], dim=1)

        out = out + other
        out = self.prelu_out(out)
        return out


# ---------------------------------------------------------------------------
# Full ENet encoder with Cerberus-compatible wrapper
# ---------------------------------------------------------------------------
class ENetEncoder(nn.Module):
    """Pure ENet encoder (no classifier). Returns 3 feature maps at strides 2/4/8."""
    def __init__(self, dropout_prob=0.01):
        super().__init__()
        self.initial = InitialBlock(3, 16)

        # stage 1: 1 downsampling + 4 regular, 16 -> 64
        self.stage1_0 = DownsamplingBottleneck(16, 64, dropout_prob=dropout_prob)
        self.stage1_1 = RegularBottleneck(64, dropout_prob=dropout_prob)
        self.stage1_2 = RegularBottleneck(64, dropout_prob=dropout_prob)
        self.stage1_3 = RegularBottleneck(64, dropout_prob=dropout_prob)
        self.stage1_4 = RegularBottleneck(64, dropout_prob=dropout_prob)

        # stage 2: 1 downsampling + 8 regular (with dilated/asymmetric as in paper), 64 -> 128
        self.stage2_0 = DownsamplingBottleneck(64, 128, dropout_prob=dropout_prob)
        self.stage2_1 = RegularBottleneck(128, dropout_prob=dropout_prob)
        self.stage2_2 = RegularBottleneck(128, dropout_prob=dropout_prob, dilation=2)
        self.stage2_3 = RegularBottleneck(128, dropout_prob=dropout_prob, asymmetric=True)
        self.stage2_4 = RegularBottleneck(128, dropout_prob=dropout_prob, dilation=4)
        self.stage2_5 = RegularBottleneck(128, dropout_prob=dropout_prob)
        self.stage2_6 = RegularBottleneck(128, dropout_prob=dropout_prob, dilation=8)
        self.stage2_7 = RegularBottleneck(128, dropout_prob=dropout_prob, asymmetric=True)
        self.stage2_8 = RegularBottleneck(128, dropout_prob=dropout_prob, dilation=16)

        # stage 3: 8 regular (no downsampling), keep 128
        self.stage3_0 = RegularBottleneck(128, dropout_prob=dropout_prob)
        self.stage3_1 = RegularBottleneck(128, dropout_prob=dropout_prob, dilation=2)
        self.stage3_2 = RegularBottleneck(128, dropout_prob=dropout_prob, asymmetric=True)
        self.stage3_3 = RegularBottleneck(128, dropout_prob=dropout_prob, dilation=4)
        self.stage3_4 = RegularBottleneck(128, dropout_prob=dropout_prob)
        self.stage3_5 = RegularBottleneck(128, dropout_prob=dropout_prob, dilation=8)
        self.stage3_6 = RegularBottleneck(128, dropout_prob=dropout_prob, asymmetric=True)
        self.stage3_7 = RegularBottleneck(128, dropout_prob=dropout_prob, dilation=16)

        # for stage-wise init compatibility
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ENet(nn.Module):
    """Cerberus-compatible ENet wrapper.

    Returns [x0(stride1, 16ch), x1(stride2,16ch), x2(stride4,64ch),
             x3(stride8,128ch), x4(stride16,128ch)].
    """
    def __init__(self, dropout_prob=0.01):
        super().__init__()
        self.encoder = ENetEncoder(dropout_prob=dropout_prob)
        # synthesize stride16 from stride8 via strided conv
        self.down_to_16 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
        )
        self.filter_info = [16, 16, 64, 128, 128]

    def forward(self, x):
        # initial: stride 2 -> x1
        x1 = self.encoder.initial(x)  # (B,16,H/2,W/2)

        # stage1: stride 4 -> x2
        x = self.encoder.stage1_0(x1)
        x = self.encoder.stage1_1(x)
        x = self.encoder.stage1_2(x)
        x = self.encoder.stage1_3(x)
        x = self.encoder.stage1_4(x)
        x2 = x  # (B,64,H/4,W/4)

        # stage2: stride 8 -> x3
        x = self.encoder.stage2_0(x)
        x = self.encoder.stage2_1(x)
        x = self.encoder.stage2_2(x)
        x = self.encoder.stage2_3(x)
        x = self.encoder.stage2_4(x)
        x = self.encoder.stage2_5(x)
        x = self.encoder.stage2_6(x)
        x = self.encoder.stage2_7(x)
        x = self.encoder.stage2_8(x)

        # stage3: keep stride 8
        x = self.encoder.stage3_0(x)
        x = self.encoder.stage3_1(x)
        x = self.encoder.stage3_2(x)
        x = self.encoder.stage3_3(x)
        x = self.encoder.stage3_4(x)
        x = self.encoder.stage3_5(x)
        x = self.encoder.stage3_6(x)
        x = self.encoder.stage3_7(x)
        x3 = x  # (B,128,H/8,W/8)

        # x4: stride 16 via extra downsample
        x4 = self.down_to_16(x3)  # (B,128,H/16,W/16)

        # x0: stride 1 via upsample of x1
        x0 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)

        return [x0, x1, x2, x3, x4]


def enet(pretrained=False, **kwargs):
    """Factory for Cerberus. `pretrained` is ignored (no ImageNet weights for ENet);
    kept for API compatibility with get_backbone(..., pretrained=...)."""
    # ENet has no standard ImageNet weights in torchvision; could load from
    # https://github.com/e-lab/ENet if needed. For now random init.
    dropout_prob = kwargs.pop('dropout_prob', 0.01)
    model = ENet(dropout_prob=dropout_prob)
    # pretrained hook: if a local checkpoint is provided via kwargs['weights_path'], load it
    weights_path = kwargs.pop('weights_path', None)
    if pretrained and weights_path is not None:
        import os
        if os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(sd, strict=False)
    return model


# alias for get_backbone compatibility
def ENet_backbone(pretrained=False, **kwargs):
    return enet(pretrained=pretrained, **kwargs)
