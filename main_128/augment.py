import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1.Basic geometric transformations --------------------------
def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)

    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return torch.tensor(rows, device=device, dtype=torch.float32)

    ref_tensor = ref[0]
    device = ref_tensor.device if device is None else device
    ref_shape = ref_tensor.shape
    processed_elems = []
    for x in elems:
        if isinstance(x, torch.Tensor):
            expanded = x.expand_as(ref_tensor)
        else:
            x_tensor = torch.tensor(x, device=device, dtype=torch.float32)
            expanded = x_tensor.expand(ref_shape)
        processed_elems.append(expanded)

    matrix_tensor = torch.stack(processed_elems, dim=-1).reshape(ref_shape + (len(rows), -1))
    matrix_tensor = matrix_tensor.squeeze(dim=tuple(range(1, len(ref_shape))))

    return matrix_tensor


def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1], **kwargs)


def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1], **kwargs)


def rotate_x(theta, **kwargs):
    return matrix(
        [1, 0, 0, 0],
        [0, torch.cos(theta), -torch.sin(theta), 0],
        [0, torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 0, 1], **kwargs)



def rotate_y(theta, **kwargs):
    return matrix(
        [torch.cos(theta), 0, torch.sin(theta), 0],
        [0, 1, 0, 0],
        [-torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1], **kwargs)


def rotate_z(theta, **kwargs):
    return matrix(
        [torch.cos(theta), -torch.sin(theta), 0, 0],
        [torch.sin(theta), torch.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1], **kwargs)


def translate3d_inv(tx, ty, tz, **kwargs):
    return translate3d(-tx, -ty, -tz, **kwargs)


def scale3d_inv(sx, sy, sz, **kwargs):
    return scale3d(1 / sx, 1 / sy, 1 / sz, **kwargs)


def rotate_x_inv(theta, **kwargs):
    return rotate_x(-theta, **kwargs)


def rotate_y_inv(theta, **kwargs):
    return rotate_y(-theta, **kwargs)


def rotate_z_inv(theta, **kwargs):
    return rotate_z(-theta, **kwargs)

# -------------------------- 2.Morphological transformation --------------------------
def erode_cross_2d(x, kernel_size):
    """Vectorized 2D cross erosion for a batch of slices [N, C, H, W]."""
    padding = kernel_size // 2
    x_h = F.pad(x, (padding, padding, 0, 0), mode='reflect')
    x_v = F.pad(x, (0, 0, padding, padding), mode='reflect')

    min_h = -F.max_pool2d(-x_h, kernel_size=(1, kernel_size), stride=1)
    min_v = -F.max_pool2d(-x_v, kernel_size=(kernel_size, 1), stride=1)

    return torch.minimum(min_h, min_v)


def dilate_cross_2d(x, kernel_size):
    """Vectorized 2D cross dilation for a batch of slices [N, C, H, W]."""
    padding = kernel_size // 2
    x_h = F.pad(x, (padding, padding, 0, 0), mode='reflect')
    x_v = F.pad(x, (0, 0, padding, padding), mode='reflect')
    max_h = F.max_pool2d(x_h, kernel_size=(1, kernel_size), stride=1)
    max_v = F.max_pool2d(x_v, kernel_size=(kernel_size, 1), stride=1)

    return torch.maximum(max_h, max_v)


# -------------------------- 3.Gaussian perturbation --------------------------
def get_pore_boundary_mask_3d(volumes: torch.Tensor, kernel_size=3) -> torch.Tensor:
    batch_size, channels, depth, height, width = volumes.shape
    device = volumes.device
    padding = kernel_size // 2

    kernel = torch.zeros((1, 1, kernel_size, kernel_size, kernel_size), device=device)
    center = kernel_size // 2
    kernel[0, 0, center, center, :] = 1.0  # W
    kernel[0, 0, center, :, center] = 1.0  # H
    kernel[0, 0, :, center, center] = 1.0  # D

    pore_mask = (volumes <= 0).float()
    solid_in_neighbor = F.conv3d(
        1 - pore_mask,
        kernel,
        padding=padding,
        groups=channels
    )
    boundary_mask = (pore_mask * (solid_in_neighbor > 0)).float()

    return boundary_mask


def gaussian_perturb_3d(
    volumes: torch.Tensor,
    boundary_mask: torch.Tensor,
    sigma_boundary: float = 0.09,
    sigma_interior: float = 0.04
) -> torch.Tensor:

    device = volumes.device
    gaussian_noise = torch.randn_like(volumes, device=device)

    sigma_tensor = sigma_interior * torch.ones_like(volumes, device=device)
    sigma_tensor = torch.where(
        boundary_mask == 1,
        sigma_boundary * torch.ones_like(sigma_tensor),
        sigma_tensor
    )

    scaled_noise = gaussian_noise * sigma_tensor
    scaled_noise = torch.clamp(scaled_noise, min=-2*sigma_tensor, max=2*sigma_tensor)

    perturbed = volumes + scaled_noise
    perturbed = torch.where(
        volumes <= 0,
        torch.clamp(perturbed, min=-1.0, max=0.0),
        torch.clamp(perturbed, min=0.0, max=1.0)
    )

    return perturbed


def numeric_perturb_3d(
    volumes: torch.Tensor,
    sigma_boundary: float = 0.09,
    sigma_interior: float = 0.04
) -> torch.Tensor:

    boundary_mask = get_pore_boundary_mask_3d(volumes, kernel_size=3)

    perturbed = gaussian_perturb_3d(
        volumes=volumes,
        boundary_mask=boundary_mask,
        sigma_boundary=sigma_boundary,
        sigma_interior=sigma_interior
    )
    return perturbed


#######################################################################################
class AugmentPipe3D(nn.Module):
    def __init__(self,
                 # 1.geometric transformations
                 xflip=0.6,
                 yflip=0.6,
                 zflip=0.6,
                 rotate90=0.6,
                 # 2.Morphological transformation
                 erode_prob=0.3,
                 dilate_prob=0.3,
                 morph_kernel_size=3,
                 # 3.Gaussian perturbation
                 numeric_perturb_prob: float = 0.4,
                 sigma_boundary: float = 0.09,
                 sigma_interior: float = 0.04
                 ):
        super().__init__()

        self.register_buffer('p', torch.tensor(0.0))

        self.xflip = float(xflip)
        self.yflip = float(yflip)
        self.zflip = float(zflip)
        self.rotate90 = float(rotate90)

        self.register_buffer('ada_stats', torch.tensor(0.0))
        self.register_buffer('ada_steps', torch.tensor(0))

        self.erode_prob = erode_prob
        self.dilate_prob = dilate_prob
        self.morph_kernel_size = morph_kernel_size

        self.numeric_perturb_prob = numeric_perturb_prob
        self.sigma_boundary = sigma_boundary
        self.sigma_interior = sigma_interior


    def update_ada_stats(self, real_pred):

        signal = torch.sign(real_pred).mean().detach()
        self.ada_stats.mul_(0.99).add_(signal, alpha=0.01)
        self.ada_steps.add_(1)


    def adjust_p(self, target=0.6, rate=0.001):

        if self.ada_steps.item() < 10:
            return None, self.p.item()

        current_signal = self.ada_stats.item()
        delta = (target - current_signal) * rate

        new_p = self.p.item() - delta
        new_p = max(0.0, min(1.0, new_p))
        self.p.copy_(torch.tensor(new_p))

        return current_signal, new_p


    def forward(self, volumes):

        assert isinstance(volumes, torch.Tensor) and volumes.ndim == 5, "The input must be a 5D tensor [B, C, D, H, W]"
        batch_size, num_channels, depth, height, width = volumes.shape
        device = volumes.device

        I_4 = torch.eye(4, device=device)
        G_inv = I_4.expand(batch_size, 4, 4)

        # X-axis inversion: Probability is (xflip * p)
        if self.xflip > 0:
            flip_mask = (torch.rand([batch_size], device=device) < self.xflip * self.p).float()
            scale = 1 - 2 * flip_mask[:, None, None]
            flip_matrix = scale3d_inv(scale, torch.ones_like(scale), torch.ones_like(scale), device=device)
            G_inv = G_inv @ flip_matrix

        # Y-axis inversion: Probability is (yflip * p)
        if self.yflip > 0:
            flip_mask = (torch.rand([batch_size], device=device) < self.yflip * self.p).float()
            scale = 1 - 2 * flip_mask[:, None, None]
            flip_matrix = scale3d_inv(torch.ones_like(scale), scale, torch.ones_like(scale), device=device)
            G_inv = G_inv @ flip_matrix

        # Z-axis inversion: Probability is (zflip * p)
        if self.zflip > 0:
            flip_mask = (torch.rand([batch_size], device=device) < self.zflip * self.p).float()
            scale = 1 - 2 * flip_mask[:, None, None]
            flip_matrix = scale3d_inv(torch.ones_like(scale), torch.ones_like(scale), scale, device=device)
            G_inv = G_inv @ flip_matrix

        # 90-degree rotation: Randomly select one of the three axes for rotation.
        if self.rotate90 > 0:
            rotate_mask = (torch.rand([batch_size], device=device) < self.rotate90 * self.p)
            rotations = torch.where(rotate_mask,
                                    torch.randint(1, 4, [batch_size], device=device),
                                    torch.zeros([batch_size], device=device, dtype=torch.int32))

            axes = torch.randint(0, 3, [batch_size], device=device)  # 0:X, 1:Y, 2:Z

            for i in range(batch_size):
                if rotate_mask[i]:
                    theta = rotations[i].float() * (torch.pi / 2)
                    if axes[i] == 0:
                        rot_matrix = rotate_x_inv(theta, device=device)
                    elif axes[i] == 1:
                        rot_matrix = rotate_y_inv(theta, device=device)
                    else:
                        rot_matrix = rotate_z_inv(theta, device=device)
                    G_inv[i] = G_inv[i] @ rot_matrix

        ############################# 1.geometric transformations #############################
        if not torch.allclose(G_inv, I_4.expand_as(G_inv), atol=1e-6):

            grid = F.affine_grid(G_inv[:, :3, :], volumes.shape, align_corners=False)
            volumes = F.grid_sample(volumes, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

        ############################# 2. Morphological transformation #############################
        if (self.erode_prob > 0 or self.dilate_prob > 0) and self.p > 0:

            total_morph_prob = (self.erode_prob + self.dilate_prob) * self.p
            morph_mask = torch.rand(batch_size, device=device) < total_morph_prob

            if morph_mask.any():

                total_prob = self.erode_prob + self.dilate_prob
                erode_weight = self.erode_prob / total_prob if total_prob > 0 else 0.5

                volumes_out = volumes.clone()
                selected_idx = torch.nonzero(morph_mask, as_tuple=False).squeeze(1)
                selected_count = selected_idx.numel()

                op_is_erode = torch.rand(selected_count, device=device) < erode_weight

                if op_is_erode.any():
                    erode_idx = selected_idx[op_is_erode]
                    erode_vol = volumes[erode_idx]

                    erode_flat = erode_vol.permute(0, 2, 1, 3, 4).reshape(-1, num_channels, height, width)
                    erode_flat = erode_cross_2d(erode_flat, self.morph_kernel_size)

                    erode_vol = erode_flat.reshape(erode_idx.numel(), depth, num_channels, height, width).permute(
                        0, 2, 1, 3, 4
                    )
                    volumes_out[erode_idx] = erode_vol

                if (~op_is_erode).any():
                    dilate_idx = selected_idx[~op_is_erode]
                    dilate_vol = volumes[dilate_idx]
                    dilate_flat = dilate_vol.permute(0, 2, 1, 3, 4).reshape(-1, num_channels, height, width)
                    dilate_flat = dilate_cross_2d(dilate_flat, self.morph_kernel_size)
                    dilate_vol = dilate_flat.reshape(dilate_idx.numel(), depth, num_channels, height, width).permute(
                        0, 2, 1, 3, 4
                    )
                    volumes_out[dilate_idx] = dilate_vol

                volumes = volumes_out

        ############################# 3.Gaussian perturbation #############################
        if self.numeric_perturb_prob > 0 and self.p > 0:
            current_numeric_prob = self.numeric_perturb_prob * self.p
            numeric_mask = (torch.rand([batch_size], device=device) < current_numeric_prob).float()
            if numeric_mask.any():

                numeric_mask_3d = numeric_mask[:, None, None, None, None].expand_as(volumes)

                perturbed_volumes = numeric_perturb_3d(
                    volumes=volumes,
                    sigma_boundary=self.sigma_boundary,
                    sigma_interior=self.sigma_interior
                    )

                volumes = volumes * (1 - numeric_mask_3d) + perturbed_volumes * numeric_mask_3d

        return volumes
