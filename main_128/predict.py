"""
Prediction script for generating 3D volumes from a trained StyleGAN model.
Supports batch generation, binarization, and saving as .raw files and PNG slices.
"""

import os
import time
import torch
import numpy as np
import torchvision
import math
from network import Generator, MappingNetwork


def load_model(checkpoint_path, device):
    """
    Loads a trained generator and mapping network from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        tuple: (generator, mapping_network, d_latent)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    image_size = 128
    log_resolution = int(math.log2(image_size))
    d_latent = 128
    mapping_layers = 4

    generator = Generator(log_resolution, d_latent).to(device)
    mapping_network = MappingNetwork(d_latent, mapping_layers).to(device)

    generator.load_state_dict(checkpoint['generator'])
    mapping_network.load_state_dict(checkpoint['mapping_network'])

    generator.eval()
    mapping_network.eval()

    return generator, mapping_network, d_latent


def generate_noise(batch_size, n_blocks, device):
    """
    Generates a list of 3D noise tensors for each generator block.

    Args:
        batch_size (int): Number of samples in the batch.
        n_blocks (int): Number of generator blocks.
        device (torch.device): Device to place tensors on.

    Returns:
        list: List of tuples (noise1, noise2) for each block.
    """
    noise = []
    resolution = 4
    for i in range(n_blocks):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, resolution, device=device)
        noise.append((n1, n2))
        resolution *= 2

    return noise


def generate_latents(batch_size, d_latent, mapping_network, device, generator):
    """
    Generates style vectors w from random latent codes z using the mapping network.

    Args:
        batch_size (int): Number of samples.
        d_latent (int): Dimensionality of the latent space.
        mapping_network (MappingNetwork): The mapping network.
        device (torch.device): Device to use.
        generator (Generator): The generator (used to determine number of blocks).

    Returns:
        torch.Tensor: Style vectors w with shape (n_blocks, batch_size, d_latent).
    """
    z = torch.randn(batch_size, d_latent, device=device)
    w = mapping_network(z)
    n_blocks = generator.n_blocks

    return w[None, :, :].expand(n_blocks, -1, -1)


def binarize_volume(volume, threshold=0.0):
    """
    Binarizes a volume by thresholding values.

    Args:
        volume (torch.Tensor): Input volume.
        threshold (float): Threshold value.

    Returns:
        torch.Tensor: Binarized volume (values 0 or 1).
    """
    return (volume > threshold).float()


def save_raw(volume, path):
    """
    Saves a volume as a .raw binary file (uint8).

    Args:
        volume (torch.Tensor): Volume tensor of shape (1, D, H, W) or (D, H, W).
        path (str): Output file path.
    """
    vol_np = volume.squeeze().cpu().numpy()
    vol_np = (vol_np * 255).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(vol_np.tobytes())


def generate_and_save(checkpoint_path, output_dir, num_samples=100, batch_size=6):
    """
    Generates volumes and saves them as .raw files and PNG grid images.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        output_dir (str): Root output directory.
        num_samples (int): Total number of volumes to generate.
        batch_size (int): Batch size for generation.
    """
    raw_dir = os.path.join(output_dir, 'raw_volumes')
    png_dir = os.path.join(output_dir, 'png_slices_z32')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator, mapping_network, d_latent = load_model(checkpoint_path, device)
    n_blocks = generator.n_blocks

    total_generated = 0
    grid_idx = 0

    with torch.no_grad():
        while total_generated < num_samples:
            current_batch_size = min(batch_size, num_samples - total_generated)

            w = generate_latents(current_batch_size, d_latent, mapping_network, device, generator)
            noise = generate_noise(current_batch_size, n_blocks, device)

            volumes = generator(w, noise)

            binarized_volumes = binarize_volume(volumes, threshold=0.0)

            for i in range(current_batch_size):
                vol_idx = total_generated + i
                raw_path = os.path.join(raw_dir, f'volume_{vol_idx:06d}.raw')
                save_raw(binarized_volumes[i], raw_path)

            # Save a grid of z=32 slices if batch is full
            if current_batch_size == batch_size:
                z32_slices = [binarized_volumes[i, :, :, 31].unsqueeze(0) for i in range(current_batch_size)]
                z32_slices = torch.cat(z32_slices, dim=0)

                grid = torchvision.utils.make_grid(
                    z32_slices,
                    nrow=3,
                    normalize=True,
                    padding=2
                )

                img = torchvision.transforms.ToPILImage()(grid.cpu())
                grid_path = os.path.join(png_dir, f'grid_z32_{grid_idx:04d}.png')
                img.save(grid_path)
                grid_idx += 1

            total_generated += current_batch_size


if __name__ == '__main__':
    start_time = time.time()

    CHECKPOINT_PATH = 'results_Castlegate/checkpoints/ckpt_step_26600.pth'
    OUTPUT_DIR = 'results_Castlegate/generated_volume_2.66w'
    NUM_SAMPLES = 300
    BATCH_SIZE = 6

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created the main output directory: {OUTPUT_DIR}")
    else:
        print(f"The main output directory already exists.: {OUTPUT_DIR}")

    generate_and_save(
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE
    )

    time_elapsed = (time.time() - start_time)
    print(f'Total duration: {time_elapsed:.2f} s')