import math
from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

class MappingNetwork(nn.Module):
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)

        return self.net(z)


class Generator(nn.Module):
    """4×4×4 => 128×128×128"""

    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 4, max_features: int = 128):
        super().__init__()
        # log_resolution-2 = 7-2 = 5     i range( 5,4,3,2,1,0)
        # min( 128 , 4*(2 ** i) )  =>  [128，64，32，16，8，4]
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4, 4)))

        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_volume = ToVolume(d_latent, features[0])

        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.up_sample = UpSample3D()
        self.activation = nn.Tanh()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):

        batch_size = w.shape[1]
        x = self.initial_constant.expand(batch_size, -1, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        volume = self.to_volume(x, w[0])

        for i in range(1, self.n_blocks):
            x = self.up_sample(x)
            x, volume_new = self.blocks[i - 1](x, w[i], input_noise[i])
            volume = self.up_sample(volume) + volume_new

        return self.activation(volume)


class GeneratorBlock(nn.Module):

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.to_volume = ToVolume(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        volume = self.to_volume(x, w)

        return x, volume


class StyleBlock(nn.Module):

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv3dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        s = self.to_style(w)
        x = self.conv(x, s)

        if noise is not None:
            x = x + self.scale_noise[None, :, None, None, None] * noise

        return self.activation(x + self.bias[None, :, None, None, None])


class ToVolume(nn.Module):

    def __init__(self, d_latent: int, features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        self.conv = Conv3dWeightModulate(features, 1, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        style = self.to_style(w)
        x = self.conv(x, style)

        return self.activation(x + self.bias[None, :, None, None, None])


class Conv3dWeightModulate(nn.Module):

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: bool = True, eps: float = 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        b, _, d, h, w = x.shape
        s = s[:, None, :, None, None, None]
        weights = self.weight()[None, :, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, d, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)
        x = F.conv3d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, d, h, w)


class Discriminator(nn.Module):

    def __init__(self, log_resolution: int, n_features: int = 4, max_features: int = 128):
        super().__init__()

        self.from_volume = nn.Sequential(
            EqualizedConv3d(1, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        self.blocks = nn.Sequential(*[DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)])

        self.std_dev = MiniBatchStdDev3D()
        final_features = features[-1] + 33

        self.conv = EqualizedConv3d(final_features, final_features, 3, padding=1)
        self.final = EqualizedLinear(4 * 4 * 4 * final_features, 1)  # 适配3D最终尺寸


        self.std_conv_layers = nn.Sequential(
            EqualizedConv3d(1, 2, kernel_size=3, stride=2, padding=1),
            EqualizedConv3d(2, 4, kernel_size=3, stride=2, padding=1),
            EqualizedConv3d(4, 8, kernel_size=3, stride=2, padding=1),
            EqualizedConv3d(8, 16, kernel_size=3, stride=2, padding=1),
            EqualizedConv3d(16, 32, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x: torch.Tensor):
        x = self.from_volume(x)
        _, std = self.std_dev(x)
        std = self.std_conv_layers(std)

        x = self.blocks(x)
        x, _ = self.std_dev(x)

        x = torch.cat([x, std], dim=1)
        x = self.conv(x)

        x = x.reshape(x.shape[0], -1)

        return self.final(x)


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(
            DownSample3D(),
            EqualizedConv3d(in_features, out_features, kernel_size=1)
        )
        self.block = nn.Sequential(
            EqualizedConv3d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv3d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.down_sample = DownSample3D()
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


class MiniBatchStdDev3D(nn.Module):

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):

        G = self.group_size
        N, C, D, H, W = x.shape
        assert N % G == 0, "The batch size must be an integer multiple of group_size."
        M = N // G

        grouped = x.view(G, M, C, D, H, W)
        mean = grouped.mean(dim=0, keepdim=True)  # [1, M, C, D, H, W]
        grouped = grouped - mean
        var = grouped.var(dim=0, unbiased=False)  # [M, C, D, H, W]
        std = torch.sqrt(var + 1e-8)  # [M, C, D, H, W]
        std = std.mean(dim=[1, 2, 3, 4], keepdim=True)  # [M, 1, 1, 1, 1]
        std = std.repeat_interleave(G, dim=0)  # [N, 1, 1, 1, 1]
        std = std.expand(-1, -1, D, H, W)  # [N, 1, D, H, W]

        return torch.cat([x, std], dim=1), std  # [N, C+1, D, H, W]


class DownSample3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.smooth = Smooth3D()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)

        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2, x.shape[4] // 2),
                             mode='trilinear', align_corners=False)


class UpSample3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.smooth = Smooth3D()

    def forward(self, x: torch.Tensor):

        return self.smooth(self.up_sample(x))


class Smooth3D(nn.Module):

    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[[[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                                 [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
                                 [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad3d(1)

    def forward(self, x: torch.Tensor):
        b, c, d, h, w = x.shape
        x = x.view(-1, 1, d, h, w)
        x = self.pad(x)
        x = F.conv3d(x, self.kernel)

        return x.view(b, c, d, h, w)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):

        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv3d(nn.Module):

    def __init__(self, in_features: int, out_features: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):

        return F.conv3d(x, self.weight(), bias=self.bias, stride=self.stride, padding=self.padding)


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):

        return self.weight * self.c



