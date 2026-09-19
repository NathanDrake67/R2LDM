import time
import numpy as np
import math

import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer

@NECKS.register_module
class RPN(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):

            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks).to('cuda')
        self.deblocks = nn.ModuleList(deblocks).to('cuda')

        #logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            if j < num_blocks -1:
                block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = F.relu(self.blocks[i](x))
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return x

@NECKS.register_module
class S2D_RPN(RPN):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(S2D_RPN, self).__init__(layer_nums,ds_layer_strides,ds_num_filters,us_layer_strides,us_num_filters,num_input_features,norm_cfg, name,logger)

        # S2D module

        self.encoder_1_new = Sequential(
            Conv2d(8, 32, 2, 2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        ).to('cuda')

        self.encoder_2_new = Sequential(
            Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        ).to('cuda')

        self.convnext_block_1_hmq = Sequential(
            nn.Conv2d(32, 32, kernel_size=7, padding=3, groups=32),
            nn.LayerNorm([32, 20, 20], eps=1e-6),
            nn.Conv2d(32, 32 * 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, 0),
        ).to('cuda')
        self.convnext_block_2_hmq = Sequential(
            nn.Conv2d(32, 32, kernel_size=7, padding=3, groups=32),
            nn.LayerNorm([32, 20, 20], eps=1e-6),
            nn.Conv2d(32, 32 * 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, 0),
        ).to('cuda')

        self.convnext_block_3_hmq = Sequential(
            nn.Conv2d(32, 32, kernel_size=7, padding=3, groups=32),
            nn.LayerNorm([32, 20, 20], eps=1e-6),
            nn.Conv2d(32, 32 * 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, 0),
        ).to('cuda')

        self.decoder_1_new = Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        ).to('cuda')

        self.decoder_2_new = Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        ).to('cuda')
        
        self.fusion_sparse = Sequential(
            nn.Conv2d(num_input_features,num_input_features,1,1,0),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        ).to('cuda')

        self.fusion_dense = Sequential(
            nn.Conv2d(num_input_features,num_input_features,1,1,0),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        ).to('cuda')

        self.out_conv_new = Sequential(
            nn.Conv2d(128, 640, 1, 1, 0),
            nn.BatchNorm2d(640),
            nn.GELU(),
        ).to('cuda')

        # logger.info("Finish S2D Initialization")

        self.generator_1 = Sequential( # N,128,5,188,188
            nn.Conv3d(128,32,1,1,0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32,32,4,2,1), # N,32,10,376,376
            nn.BatchNorm3d(32),
            nn.ReLU(),
        ).to('cuda')

        self.gen_out_4_old = Sequential(
            nn.Conv3d(32,3,1,1,0),
        ).to('cuda')
        self.gen_out_4 = Sequential(
            nn.Conv3d(32, 16, 1, 1, 0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, 1, 1, 0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 3, 1, 1, 0),
            nn.BatchNorm3d(3),
            nn.ReLU(),
        ).to('cuda')

        self.gen_mask_4_old = Sequential(
            nn.Conv3d(32,2,1,1,0),
        ).to('cuda')

        self.gen_mask_4 = Sequential(
            nn.Conv3d(32, 16, 1, 1, 0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, 1, 1, 0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1, 1, 0),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        ).to('cuda')

        self.gen_mask_4_two = Sequential(
            nn.Conv3d(32, 16, 1, 1, 0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, 1, 1, 0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 2, 1, 1, 0),
            nn.BatchNorm3d(2),
            nn.ReLU(),
        ).to('cuda')

        self.generator_2 = Sequential(
            nn.Conv3d(32,16,1,1,0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16,16,4,2,1), # N,16,20,752,752
            nn.BatchNorm3d(16),
            nn.ReLU(),
        ).to('cuda')

        self.gen_out_2 = Sequential(
            nn.Conv3d(16,3,1,1,0),
        ).to('cuda')

        self.gen_mask_2 = Sequential(
            nn.Conv3d(16,1,1,1,0),
        ).to('cuda')
        self.gen_mask_2_two = Sequential(
            nn.Conv3d(16, 2, 1, 1, 0),
        ).to('cuda')
        # logger.info("Finish PCR Initialization")
        # 0914
        self.generator_3 = Sequential(
            nn.Conv3d(16, 8, 1, 1, 0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 3, 4, 2, 1),  # N,16,20,752,752
            nn.BatchNorm3d(3),
            nn.ReLU(),
        ).to('cuda')

        self.gen_out_1 = Sequential(
            nn.Conv3d(3, 3, 1, 1, 0),
        ).to('cuda')

        self.gen_mask_1 = Sequential(
            nn.Conv3d(3, 1, 1, 1, 0),
        ).to('cuda')
        self.gen_mask_1_two = Sequential(
            nn.Conv3d(3, 2, 1, 1, 0),
        ).to('cuda')


    def forward(self, x):
        # S2D Module
        y_1 = self.encoder_1_new(x)
        y_2 = self.encoder_2_new(y_1)
        att = self.convnext_block_1_hmq(y_2) + y_2
        att = self.convnext_block_2_hmq(att) + att
        att = F.gelu(self.convnext_block_3_hmq(att) + att)
        y_3 = torch.cat([self.decoder_1_new(att), y_1], 1)
        F_S_b = self.decoder_2_new(y_3)
        F_S_a = F_S_b

        # PCR Module
        batch_size, _, height, width = y_1.shape
        gen = self.out_conv_new(F_S_b)
        gen = gen.view(batch_size, 128, 5, height, width)
        gen = self.generator_1(gen)
        gen_offset_4 = self.gen_out_4(gen)
        gen_mask_4 = self.gen_mask_4_two(gen)
        gen = self.generator_2(gen)
        gen_mask_2 = self.gen_mask_2_two(gen)
        gen_offset_2 = self.gen_out_2(gen)
        gen = self.generator_3(gen)
        gen_mask_1 = self.gen_mask_1_two(gen)
        gen_offset_1 = self.gen_out_1(gen)

        return x, gen_offset_2, gen_mask_2, gen_offset_1, gen_mask_1, F_S_a, F_S_b
