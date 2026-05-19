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
        z = F.normalize(z, dim=1)  # 归一化
        return self.net(z)


class Generator(nn.Module):
    """3D生成器：从4×4×4常量体积逐步上采样到64×64×64"""
    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):
        super().__init__()
        # log_resolution-2 = 6-2 = 4     i range( 4,3,2,1,0)
        # 计算各块的特征数  min( 512 , 32*(2 ** i) )  =>  [512, 256, 128, 64, 32]
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)   # 5

        # 3D初始常量 (1, 512, 4, 4, 4)
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4, 4)))

        # 第一个风格块（4×4×4）和3D体积转换层
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_volume = ToVolume(d_latent, features[0])  # 替换ToRGB为ToVolume（单通道3D）

        # 生成器块（逐步上采样到64×64×64）
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # 3D上采样层
        self.up_sample = UpSample3D()

        self.activation = nn.Tanh()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        batch_size = w.shape[1]
        # 扩展初始常量到批次大小（3D）  (3, 512, 4, 4, 4)
        x = self.initial_constant.expand(batch_size, -1, -1, -1, -1)

        # 第一个风格块
        x = self.style_block(x, w[0], input_noise[0][1])
        # 初始3D体积
        volume = self.to_volume(x, w[0])

        # 逐步上采样到目标分辨率
        for i in range(1, self.n_blocks):
            x = self.up_sample(x)  # 3D上采样
            x, volume_new = self.blocks[i - 1](x, w[i], input_noise[i])
            volume = self.up_sample(volume) + volume_new  # 累加各层体积

        return self.activation(volume)


class GeneratorBlock(nn.Module):
    """3D生成器块：包含两个3D风格块和3D体积输出"""
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.to_volume = ToVolume(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        x = self.style_block1(x, w, noise[0])  # 3D风格块1
        x = self.style_block2(x, w, noise[1])  # 3D风格块2
        volume = self.to_volume(x, w)  # 3D体积输出
        return x, volume


class StyleBlock(nn.Module):
    """3D风格块：权重调制3D卷积"""
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv3dWeightModulate(in_features, out_features, kernel_size=3)  # 3D卷积
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        s = self.to_style(w)  # 风格向量
        x = self.conv(x, s)   # 3D权重调制卷积
        # 添加3D噪声
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None, None])


class ToVolume(nn.Module):
    """3D体积转换层：从特征图生成单通道3D体积"""
    def __init__(self, d_latent: int, features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        self.conv = Conv3dWeightModulate(features, 1, kernel_size=1, demodulate=False)  # 输出单通道
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        style = self.to_style(w)
        x = self.conv(x, style)  # 3D卷积
        return self.activation(x + self.bias[None, :, None, None, None])


class Conv3dWeightModulate(nn.Module):
    """3D权重调制与去调制卷积"""
    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: bool = True, eps: float = 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2  # 3D padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size, kernel_size])  # 3D权重
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        b, _, d, h, w = x.shape  # 3D维度：depth, height, width
        s = s[:, None, :, None, None, None]  # 适配3D权重形状
        weights = self.weight()[None, :, :, :, :, :]
        weights = weights * s  # 权重调制

        # 去调制（3D）
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights **2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        # 3D分组卷积
        x = x.reshape(1, -1, d, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)
        x = F.conv3d(x, weights, padding=self.padding, groups=b)  # 3D卷积
        return x.reshape(-1, self.out_features, d, h, w)


class Discriminator(nn.Module):
    """3D判别器：处理128×128×128单通道体积"""
    def __init__(self, log_resolution: int, n_features: int = 32, max_features: int = 512):
        super().__init__()
        # 从单通道3D体积转换为特征图
        self.from_volume = nn.Sequential(
            EqualizedConv3d(1, n_features, 1),  # 3D卷积
            nn.LeakyReLU(0.2, True),
        )

        # 特征数计算（3D）
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        self.blocks = nn.Sequential(*[DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)])

        # 3D迷你批次标准差 (3, 513, 4, 4, 4)
        self.std_dev = MiniBatchStdDev3D()
        final_features = features[-1] + 1  # 513
        # # 3D卷积   新添加padding=1确保形状不变
        self.conv = EqualizedConv3d(final_features, final_features, 3, padding=1)
        # 最终特征图尺寸为 4×4×4
        self.final = EqualizedLinear(4 * 4 * 4 * final_features, 1)  # 适配3D最终尺寸

    def forward(self, x: torch.Tensor):
        x = x - 0.5  # 简单归一化   # [0,1] => [-0.5,0.5]
        x = self.from_volume(x)  # 从3D体积到特征图
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)  # 展平3D特征 （3，513*4*4*4）
        return self.final(x)           # （3，1）


class DiscriminatorBlock(nn.Module):
    """3D判别器块：3D卷积+残差连接"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(
            DownSample3D(),  # 3D下采样
            EqualizedConv3d(in_features, out_features, kernel_size=1)  # 3D卷积
        )
        self.block = nn.Sequential(
            EqualizedConv3d(in_features, in_features, kernel_size=3, padding=1),  # 3D卷积
            nn.LeakyReLU(0.2, True),
            EqualizedConv3d(in_features, out_features, kernel_size=3, padding=1),  # 3D卷积下采样
            nn.LeakyReLU(0.2, True),
        )
        self.down_sample = DownSample3D()
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        # 残差路径：下采样+1×1卷积
        residual = self.residual(x)
        # 主路径：卷积+下采样
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale   # # 融合残差与主路径


class MiniBatchStdDev3D(nn.Module):
    """3D迷你批次标准差：计算3D特征图批次内标准差"""
    # group_size不可为 1
    # 为 1时，var(dim=0) 本质上是“单个样本的方差”（理论上应为 0，但浮点误差会导致数值异常）
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        # 批次大小（batch_size）必须是 group_size 的整数倍。
        assert x.shape[0] % self.group_size == 0
        # (4, 512×4×4×4) = (3, 32768)
        grouped = x.view(self.group_size, -1)
        # grouped.var(dim=0)  计算小组内（3 个样本）每个特征的方差    再开方得到标准差  (32768,)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # 对所有特征的标准差取平均值得到一个标量    3D形状 (1, 1, 1, 1, 1)
        std = std.mean().view(1, 1, 1, 1, 1)
        b, _, d, h, w = x.shape
        std = std.expand(b, -1, d, h, w)
        return torch.cat([x, std], dim=1)   # (4, 512+1, 4, 4, 4)


class DownSample3D(nn.Module):
    """3D下采样：平滑+3D下采样"""
    def __init__(self):
        super().__init__()
        self.smooth = Smooth3D()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2, x.shape[4] // 2),
                             mode='trilinear', align_corners=False)  # 3D插值


class UpSample3D(nn.Module):
    """3D上采样：3D上采样+平滑"""
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 3D上采样
        self.smooth = Smooth3D()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


class Smooth3D(nn.Module):
    """3D平滑层：3D高斯模糊核"""
    def __init__(self):
        super().__init__()
        # 3D平滑核（1x3x3x3x1）
        kernel = torch.tensor([[[[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                                 [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
                                 [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad3d(1)  # 3D padding

    def forward(self, x: torch.Tensor):
        b, c, d, h, w = x.shape
        x = x.view(-1, 1, d, h, w)
        x = self.pad(x)
        x = F.conv3d(x, self.kernel)  # 3D卷积平滑
        return x.view(b, c, d, h, w)


# 以下类保持不变，但适配3D输入（无需修改核心逻辑）
class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedConv3d(nn.Module):
    """3D均等学习率卷积"""
    def __init__(self, in_features: int, out_features: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size, kernel_size])  # 3D权重
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv3d(x, self.weight(), bias=self.bias, padding=self.padding)  # 3D卷积


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class GradientPenalty(nn.Module):
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        batch_size = x.shape[0]
        gradients, *_ = torch.autograd.grad(outputs=d, inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta   # 指数移动平均系数 0.99
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)        #　训练步数计数器
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)    # 梯度范数的指数移动平均值

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        device = x.device
        volume_size = x.shape[2] * x.shape[3] * x.shape[4]  # 3D体积像素数 128*128*128
        # y是随机生成的 3D 体积，代表一个随机方向；
        y = torch.randn(x.shape, device=device)
        # sum()得到内积（衡量 x 在 y 方向上的投影大小）
        output = (x * y).sum() / math.sqrt(volume_size)  # 适配3D
        # 计算output（x 在随机方向 y 上的投影）对w（风格向量）的梯度   (6, 3, 512)（6个块，3个样本，512维）
        gradients, *_ = torch.autograd.grad(outputs=output, inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)
        # 路径长度（衡量w的变化对x的影响强度）    (6, 3, 512) => (6, 3)  =>  (6)
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            # a是梯度范数的历史平均值（通过指数移动平均计算），代表 “期望的平滑路径长度”；
            a = self.exp_sum_a / (1 - self.beta** self.steps)
            loss = torch.mean((norm - a) **2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        # 如beta=0.99时， 新值 = 0.99 × 旧值 + 0.01 × 当前 mean
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)
        return loss

class DiscriminatorLoss(nn.Module):
    """判别器的Logistic损失实现"""
    def __init__(self):
        super().__init__()

    def forward(self, real_output, fake_output):
        # 真实样本损失：-log(sigmoid(real_output))
        real_loss = -torch.log(torch.sigmoid(real_output) + 1e-8).mean()
        # 生成样本损失：-log(1 - sigmoid(fake_output))
        fake_loss = -torch.log(1 - torch.sigmoid(fake_output) + 1e-8).mean()
        return real_loss, fake_loss


class GeneratorLoss(nn.Module):
    """生成器的Logistic损失实现"""
    def __init__(self):
        super().__init__()

    def forward(self, fake_output):
        # 生成器损失：-log(sigmoid(fake_output))，希望生成样本被判别为真实
        gen_loss = -torch.log(torch.sigmoid(fake_output) + 1e-8).mean()
        return gen_loss

def cycle_dataloader(data_loader):
    """
    Infinite loader that recycles the data loader after each epoch
    """
    while True:
        for batch in data_loader:
            yield batch