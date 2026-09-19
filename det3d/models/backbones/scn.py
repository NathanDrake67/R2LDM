import numpy as np
import spconv.pytorch as spconv
from det3d.torchie.cnn import constant_init, kaiming_init
from det3d.torchie.trainer import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from spconv.pytorch.conv import SparseConv3d, SubMConv3d
from torch import nn

from ..registry import BACKBONES
from ..utils import build_norm_layer


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        algo=spconv.ConvAlgo.Native,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        algo=spconv.ConvAlgo.Native,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out.features = self.bn1(out.features)
        # out.features = self.relu(out.features)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        # out.features = self.bn2(out.features)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        # out.features += identity.features
        # out.features = self.relu(out.features)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


@BACKBONES.register_module
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self,
        num_input_features=128,
        norm_cfg=None,
        name="SpMiddleResNetFH",
        is_student=False,
        **kwargs,
    ):
        super().__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False
        self.is_student = is_student

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv_input_new = spconv.SparseSequential(
            spconv.SubMConv3d(
                num_input_features,
                4,
                3,
                bias=False,
                algo=spconv.ConvAlgo.Native,
                indice_key="res0",
            ).to("cuda"),
            build_norm_layer(norm_cfg, 4)[1],
            nn.ReLU(inplace=True),
        ).to("cuda")
        self.conv1_new = spconv.SparseSequential(
            SparseBasicBlock(4, 4, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(4, 4, norm_cfg=norm_cfg, indice_key="res0"),
        ).to("cuda")

        self.conv2_new = spconv.SparseSequential(
            SparseConv3d(
                4, 4, 3, 2, padding=1, bias=False, algo=spconv.ConvAlgo.Native
            ),
            build_norm_layer(norm_cfg, 4)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(4, 4, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(4, 4, norm_cfg=norm_cfg, indice_key="res1"),
        ).to("cuda")

        self.conv3_new = spconv.SparseSequential(
            SparseConv3d(
                4, 4, 3, 2, padding=1, bias=False, algo=spconv.ConvAlgo.Native
            ),
            build_norm_layer(norm_cfg, 4)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(4, 4, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(4, 4, norm_cfg=norm_cfg, indice_key="res2"),
        ).to("cuda")

        self.conv4_new = spconv.SparseSequential(
            SparseConv3d(
                4,
                4,
                3,
                (2, 1, 1),
                padding=[0, 1, 1],
                bias=False,
                algo=spconv.ConvAlgo.Native,
            ),
            build_norm_layer(norm_cfg, 4)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(4, 4, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(4, 4, norm_cfg=norm_cfg, indice_key="res3"),
        ).to("cuda")

        self.extra_conv_new = spconv.SparseSequential(
            SparseConv3d(
                4, 4, (1, 1, 1), (2, 1, 1), bias=False, algo=spconv.ConvAlgo.Native
            ),
            build_norm_layer(norm_cfg, 4)[1],
            nn.ReLU(),
        ).to("cuda")

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1])
        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        x = self.conv_input_new(x)
        x_conv1 = self.conv1_new(x)
        x_conv2 = self.conv2_new(x_conv1)
        x_conv3 = self.conv3_new(x_conv2)
        x_conv4 = self.conv4_new(x_conv3)
        output = self.extra_conv_new(x_conv4).dense()
        batch_size, channels, depth, height, width = output.shape
        output = output.view(batch_size, channels * depth, height, width)

        multi_scale_voxel_features = {
            "conv1": x_conv1,
            "conv2": x_conv2,
        }

        return output, multi_scale_voxel_features

@BACKBONES.register_module
class SpMiddleFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleFHD", **kwargs
    ):
        super(SpMiddleFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseConv3d(16, 32, 3, 2, padding=1, bias=False),  # [41, 1600, 1408] -> [21, 800, 704]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseConv3d(32, 64, 3, 2, padding=1, bias=False),  # [21, 800, 704] -> [11, 400, 352]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(64, 64, 3, 2, padding=[0, 1, 1], bias=False),  # [11, 400, 352] -> [5, 200, 176]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )
        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(64, 64, (3, 1, 1), (2, 1, 1), bias=False),  # [5, 200, 176] -> [2, 200, 176]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()

        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        conv_4 = self.middle_conv(ret)

        ret = self.extra_conv(conv_4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret, conv_4
