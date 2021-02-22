# Written by Chris Choy (cchoy@nvidia.com) 2021-02-09
import gin
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from src.models.modules.common import get_norm, get_nonlinearity, conv, block

@gin.configurable
class FCNN(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channel,
        out_channel,
        channels=(32, 48, 64, 96, 128),
        embedding_channel=1024,
        quantization_size=0.05,
        norm_type="BN",
        D=3,
    ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.quantization_size = quantization_size
        self.norm_type = norm_type
        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            norm_type=norm_type,
            D=D,
        )
        self.weight_initialization()
    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        nonlinearity_type="MinkowskiLeakyReLU",
        norm_type="BN",
        bn_momentum=0.1,
        D=3,
    ):
        self.mlp1 = nn.Sequential(
            ME.MinkowskiLinear(in_channel, channels[0], bias=False),
            get_norm(norm_type, channels[0], D=self.D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity_type)(),
        )
        self.conv1 = nn.Sequential(
            block(
                channels[0],
                channels[1],
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
                nonlinearity_type=nonlinearity_type,
                D=D,
            ),
            block(
                channels[1],
                channels[1],
                kernel_size=3,
                stride=1,
                norm_type=norm_type,
                nonlinearity_type=nonlinearity_type,
                D=D,
            ),
            ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=D),
        )
        self.conv2 = block(
            channels[1],
            channels[2],
            kernel_size=3,
            stride=2,
            norm_type=norm_type,
            nonlinearity_type=nonlinearity_type,
            D=D,
        )
        self.conv3 = block(
            channels[2],
            channels[3],
            kernel_size=3,
            stride=2,
            norm_type=norm_type,
            nonlinearity_type=nonlinearity_type,
            D=D,
        )
        self.conv4 = block(
            channels[3],
            channels[4],
            kernel_size=3,
            stride=2,
            norm_type=norm_type,
            nonlinearity_type=nonlinearity_type,
            D=D,
        )
        self.conv5 = nn.Sequential(
            conv(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel,
                kernel_size=3,
                stride=2,
                D=D,
            ),
            get_norm(norm_type, embedding_channel, D=self.D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity_type)(),
        )
        self.conv5 = nn.Sequential(
            block(
                channels[1] + channels[2] + channels[3] + channels[4],
                256,
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
                nonlinearity_type=nonlinearity_type,
                D=D,
            ),
            block(
                256,
                512,
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
                nonlinearity_type=nonlinearity_type,
                D=D,
            ),
            block(
                512,
                embedding_channel,
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
                nonlinearity_type=nonlinearity_type,
                D=D,
            ),
        )
        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.drop = ME.MinkowskiDropout()
        self.linear1 = nn.Sequential(
            ME.MinkowskiLinear(embedding_channel * 2, 512, bias=False),
            get_norm(norm_type, 512, D=self.D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity_type)(),
        )
        self.linear2 = nn.Sequential(
            ME.MinkowskiLinear(512, 256, bias=False),
            get_norm(norm_type, 256, D=self.D, bn_momentum=bn_momentum),
            get_nonlinearity(nonlinearity_type)(),
        )
        self.final = ME.MinkowskiLinear(256, out_channel, bias=True)
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, batched_coordinates: torch.Tensor, features: torch.Tensor):
        batched_coordinates[:, 1:] = batched_coordinates[:, 1:] / self.quantization_size
        x = ME.TensorField(coordinates=batched_coordinates, features=features)
        x = self.mlp1(x)
        y = x.sparse()
        y = self.conv1(y)
        y1 = self.pool(y)
        y = self.conv2(y1)
        y2 = self.pool(y)
        y = self.conv3(y2)
        y3 = self.pool(y)
        y = self.conv4(y3)
        y4 = self.pool(y)
        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)
        x = ME.cat(x1, x2, x3, x4)
        y = self.conv5(x.sparse())
        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)
        x = self.linear1(ME.cat(x1, x2))
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return self.final(x).F