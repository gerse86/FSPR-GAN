import math
from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class MappingNetwork(nn.Module):
    """ Mapping network that transforms latent code z to style vector w. """
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)  # Normalize
        return self.net(z)


class Generator(nn.Module):
    """ 3D generator: progressively upsamples from 4x4x4 constant volume to 64x64x64. """
    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):
        super().__init__()
        # log_resolution-2 = 6-2 = 4     i range( 4,3,2,1,0)
        # Compute feature maps for each block: min(512, 32*(2**i)) => [512, 256, 128, 64, 32]
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)   # 5

        # Initial 3D constant (1, 512, 4, 4, 4)
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4, 4)))

        # First style block (4x4x4) and 3D volume conversion
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_volume = ToVolume(d_latent, features[0])  # ToVolume (single-channel 3D)

        # Generator blocks (progressive upsampling to 64x64x64)
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)
        # 3D upsampling layer
        self.up_sample = UpSample3D()
        self.activation = nn.Tanh()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        batch_size = w.shape[1]
        x = self.initial_constant.expand(batch_size, -1, -1, -1, -1)

        # First style block
        x = self.style_block(x, w[0], input_noise[0][1])
        # Initial 3D volume
        volume = self.to_volume(x, w[0])

        # Progressive upsampling to target resolution
        for i in range(1, self.n_blocks):
            x = self.up_sample(x)  # 3D upsampling
            x, volume_new = self.blocks[i - 1](x, w[i], input_noise[i])
            volume = self.up_sample(volume) + volume_new  # Accumulate volumes from each layer

        return self.activation(volume)


class GeneratorBlock(nn.Module):
    """ 3D generator block with two style blocks and volume output. """
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.to_volume = ToVolume(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        x = self.style_block1(x, w, noise[0])  # 3D style block 1
        x = self.style_block2(x, w, noise[1])  # 3D style block 2
        volume = self.to_volume(x, w)  # 3D volume output
        return x, volume


class StyleBlock(nn.Module):
    """ 3D style block with weight-modulated convolution. """
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv3dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        s = self.to_style(w)  # Style vector
        x = self.conv(x, s)
        # Add 3D noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None, None])


class ToVolume(nn.Module):
    """ 3D volume conversion layer: generates single-channel volume from feature maps. """
    def __init__(self, d_latent: int, features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        self.conv = Conv3dWeightModulate(features, 1, kernel_size=1, demodulate=False)  # Output single channel
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None, None])


class Conv3dWeightModulate(nn.Module):
    """ 3D convolution with weight modulation and demodulation. """
    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: bool = True, eps: float = 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        b, _, d, h, w = x.shape  # 3D dimensions: depth, height, width
        s = s[:, None, :, None, None, None]  # Adapt to 3D weight shape
        weights = self.weight()[None, :, :, :, :, :]
        weights = weights * s  # Weight modulation

        # Demodulation
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights **2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        # 3D grouped convolution
        x = x.reshape(1, -1, d, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)
        x = F.conv3d(x, weights, padding=self.padding, groups=b)
        return x.reshape(-1, self.out_features, d, h, w)


class Discriminator(nn.Module):
    """ 3D discriminator for single-channel volumes. """
    def __init__(self, log_resolution: int, n_features: int = 32, max_features: int = 512):
        super().__init__()
        # Convert single-channel 3D volume to feature maps
        self.from_volume = nn.Sequential(
            EqualizedConv3d(1, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        self.blocks = nn.Sequential(*[DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)])
        self.std_dev = MiniBatchStdDev3D()

        final_features = features[-1] + 1
        self.conv = EqualizedConv3d(final_features, final_features, 3, padding=1)
        # Final feature map size is 4x4x4
        self.final = EqualizedLinear(4 * 4 * 4 * final_features, 1)

    def forward(self, x: torch.Tensor):
        x = x - 0.5  # Simple normalization [0,1] -> [-0.5,0.5]
        x = self.from_volume(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)


class DiscriminatorBlock(nn.Module):
    """ 3D discriminator block with convolution and residual connection. """
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
        # Residual path: downsampling + 1x1 convolution
        residual = self.residual(x)
        # Main path: convolution + downsampling
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale


class MiniBatchStdDev3D(nn.Module):
    """ 3D minibatch standard deviation: computes within-batch std across feature maps. """

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        # batch size must be divisible by group_size
        assert x.shape[0] % self.group_size == 0
        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # compute std per feature within each group, then average across all features -> scalar
        std = std.mean().view(1, 1, 1, 1, 1)
        b, _, d, h, w = x.shape
        std = std.expand(b, -1, d, h, w)
        return torch.cat([x, std], dim=1)


class DownSample3D(nn.Module):
    """ 3D downsampling with smoothing and trilinear interpolation. """
    def __init__(self):
        super().__init__()
        self.smooth = Smooth3D()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2, x.shape[4] // 2),
                             mode='trilinear', align_corners=False)


class UpSample3D(nn.Module):
    """ 3D upsampling with trilinear interpolation and smoothing. """
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.smooth = Smooth3D()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


class Smooth3D(nn.Module):
    """ 3D smoothing layer with Gaussian blur kernel. """
    def __init__(self):
        super().__init__()
        # 3D smoothing kernel (1x3x3x3x1)
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
    """ Linear layer with equalized learning rate. """
    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv3d(nn.Module):
    """ 3D convolution with equalized learning rate. """
    def __init__(self, in_features: int, out_features: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv3d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    """ Weight scaling for equalized learning rate. """
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class GradientPenalty(nn.Module):
    """ the R1 loss. """
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        batch_size = x.shape[0]
        gradients, *_ = torch.autograd.grad(outputs=d, inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm ** 2)


class DiscriminatorLoss(nn.Module):
    """ discriminator loss. """
    def __init__(self):
        super().__init__()

    def forward(self, real_output, fake_output):
        # real samples：-log(sigmoid(real_output))
        real_loss = -torch.log(torch.sigmoid(real_output) + 1e-8).mean()
        # generated samples：-log(1 - sigmoid(fake_output))
        fake_loss = -torch.log(1 - torch.sigmoid(fake_output) + 1e-8).mean()
        return real_loss, fake_loss


class GeneratorLoss(nn.Module):
    """ generator loss """
    def __init__(self):
        super().__init__()

    def forward(self, fake_output):
        # -log(sigmoid(fake_output))
        gen_loss = -torch.log(torch.sigmoid(fake_output) + 1e-8).mean()
        return gen_loss


def cycle_dataloader(data_loader):
    """ Infinite loader that recycles the data loader after each epoch """
    while True:
        for batch in data_loader:
            yield batch