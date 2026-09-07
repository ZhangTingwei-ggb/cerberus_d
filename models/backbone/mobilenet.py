from torch import nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ["MobileNetV2", "mobilenet_v2"]


model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        block=None,
    ):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super().__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        # ! HACK: holder to retrieve which layer index has down-sampling
        # ~~~~
        layer_idx = 0
        self.ds_idx_list = []
        # ~~~~

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features = [ConvBNReLU(3, input_channel, stride=1)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
                # ~~~~
                if stride != 1:
                    self.ds_idx_list.append(layer_idx)
                layer_idx += 1
                # ~~~~
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        # ~~~~ ! original
        # self.features = nn.Sequential(*features)
        # ~~~~

        # ~~~~ ! hack
        # self.old_features = nn.Sequential(*features) # for sane check
        self.features = nn.ModuleList(features)
        # ~~~~

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, input):
        # ~~~~ original
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        # x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # x = self.classifier(x)
        # ~~~~
        x = input
        feat_list = []
        for idx, layer in enumerate(self.features):
            new_x = layer(x)
            if idx in self.ds_idx_list:
                feat_list.append(x)
            x = new_x
        feat_list.append(x)  # also adding the last one

        # ~~~~ sanity check code, set strict=False when loading weight
        # assert (self.old_features(input) - x).sum() == 0
        # ~~~~
        return feat_list

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["mobilenet_v2"], progress=progress
        )
        model.load_state_dict(state_dict, strict=False)
    return model


# =============================================================================
# MobileNetV3 (torchvision based) as a Cerberus-compatible multi-scale encoder
#
# Cerberus's decoder expects a list of 5 feature maps at strides
# [16, 8, 4, 2, 1] whose channels match `filter_info`:
#     x4 (bottom, stride 16)  -> conv_map reduces to filters[-2]
#     x3 (stride 8)           -> filters[-2]
#     x2 (stride 4)           -> filters[-3]
#     x1 (stride 2)           -> filters[-4]
#     x0 (stride 1)           -> filters[-5]
#
# MobileNetV3's first conv is stride 2 and its deepest feature is stride 32,
# so we:
#   * take x1/x2/x3 from the natural feature pyramid (strides 2/4/8),
#   * take the deepest (stride 32) feature and bilinear-upsample it 2x to
#     serve as x4 at stride 16,
#   * upsample the shallowest (stride 2) feature 2x to serve as x0 at stride 1.
# =============================================================================
import torch.nn.functional as F


class MobileNetV3(nn.Module):
    """Cerberus encoder wrapper around a torchvision MobileNetV3 body.

    `features` is kept as a plain nn.Sequential so that ImageNet pretrained
    weights load directly via the usual state_dict keys (features.*).

    """

    def __init__(self, base, feature_idx_list, filter_info):
        super().__init__()
        self.features = base.features
        self.feature_idx_list = feature_idx_list
        self.filter_info = filter_info

    def forward(self, input):
        """Return [x0(stride1), x1(stride2), x2(stride4), x3(stride8), x4(stride16)]."""
        x = input
        feat_list = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.feature_idx_list:
                feat_list.append(x)
        x1, x2, x3, x4 = feat_list
        # x0: 2x bilinear upsampling of the stride-2 feature -> stride 1
        x0 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=False)
        # x4: 2x bilinear upsampling of the stride-32 feature -> stride 16
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        return [x0, x1, x2, x3, x4]


def _build_mobilenet_v3(tv_arch, pretrained, feature_idx_list, filter_info, **kwargs):
    try:
        base = tv_arch(pretrained=pretrained, **kwargs)
    except TypeError:
        # newer torchvision uses the `weights` keyword
        try:
            base = tv_arch(weights="IMAGENET1K_V1" if pretrained else None, **kwargs)
        except TypeError:
            base = tv_arch(**kwargs)
    return MobileNetV3(base, feature_idx_list, filter_info)


def mobilenet_v3_large(pretrained=False, progress=True, **kwargs):
    """MobileNetV3-Large body adapted to return Cerberus-compatible features.

    filter_info: [16, 16, 24, 40, 960]
    features:    stride 2 (16ch), stride 4 (24ch), stride 8 (40ch),
                 stride 32 (960ch) upsampled to stride 16.
    """
    from torchvision.models import mobilenet_v3_large as tv_mobilenet_v3_large

    feature_idx_list = [1, 3, 6, 16]  # 16@stride2, 24@stride4, 40@stride8, 960@stride32
    filter_info = [16, 16, 24, 40, 960]
    return _build_mobilenet_v3(
        tv_mobilenet_v3_large, pretrained, feature_idx_list, filter_info, **kwargs
    )


def mobilenet_v3_small(pretrained=False, progress=True, **kwargs):
    """MobileNetV3-Small body adapted to return Cerberus-compatible features.

    filter_info: [16, 16, 16, 24, 576]
    features:    stride 2 (16ch), stride 4 (16ch), stride 8 (24ch),
                 stride 32 (576ch) upsampled to stride 16.
    """
    from torchvision.models import mobilenet_v3_small as tv_mobilenet_v3_small

    feature_idx_list = [0, 1, 3, 12]  # 16@stride2, 16@stride4, 24@stride8, 576@stride32
    filter_info = [16, 16, 16, 24, 576]
    return _build_mobilenet_v3(
        tv_mobilenet_v3_small, pretrained, feature_idx_list, filter_info, **kwargs
    )
