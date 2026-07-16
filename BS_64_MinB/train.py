import os
import csv
import math
import time
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import numpy as np
from __init__ import Discriminator, Generator, MappingNetwork, GradientPenalty, DiscriminatorLoss, GeneratorLoss, cycle_dataloader

os.makedirs('results', exist_ok=True)
os.makedirs('results/losses', exist_ok=True)
os.makedirs('results/slices', exist_ok=True)
os.makedirs('results/checkpoints', exist_ok=True)

loss_csv_path = os.path.join('results', 'losses', 'training_losses.csv')
with open(loss_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'discriminator_loss', 'generator_loss', 'gradient_penalty',
                     'd_score_fake', 'd_score_real', 'd_score_val'])


class PorosityDataset(torch.utils.data.Dataset):
    """ Dataset for loading 3D porosity volumes from .raw files. """
    def __init__(self, path: str, volume_size: int = 64):
        super().__init__()
        self.volume_size = volume_size
        self.paths = list(Path(path).glob('*.raw'))

        if not self.paths:
            raise ValueError(f"No.raw files were found in the path {path}!")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self.paths[index]

        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)

        volume = data.reshape(1, self.volume_size, self.volume_size, self.volume_size)

        volume = torch.from_numpy(volume).float()
        volume = (volume - 0.5) / 0.5   # Normalize to [-1, 1]
        return volume


class Configs:
    """ Configuration and training logic for the StyleGAN3D model. """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.discriminator = None
        self.generator = None
        self.mapping_network = None

        self.discriminator_loss = None
        self.generator_loss = None

        self.gradient_penalty = GradientPenalty()
        self.gradient_penalty_coefficient: float = 10.

        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.mapping_network_optimizer = None

        self.loader = None

        self.batch_size: int = 20
        self.d_latent: int = 128
        self.volume_size: int = 64
        self.mapping_network_layers: int = 4
        self.learning_rate: float = 1e-3
        self.mapping_network_learning_rate: float = 1e-5
        self.gradient_accumulate_steps: int = 3
        self.adam_betas: Tuple[float, float] = (0, 0.99)
        self.style_mixing_prob: float = 0
        self.training_steps: int = 500_000
        self.n_gen_blocks: int = 0

        self.lazy_gradient_penalty_interval: int = 40

        self.log_generated_interval: int = 100
        self.save_checkpoint_interval: int = 100

        self.dataset_path: str = '../DATA/Bentheimer_64(753)'
        self.Validation_path: str = '../DATA/Bentheimer_64(80)'

        self.resume_checkpoint_path = os.path.join('results', 'checkpoints', 'ckpt_step_6300.pth')

        self.val_volumes_all = None
        self.start_step: int = 0

    def init(self):
        """ Initializes datasets, models, optimizers, and loaders. """
        dataset = PorosityDataset(self.dataset_path, self.volume_size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        self.loader = cycle_dataloader(dataloader)

        val_dataset = PorosityDataset(self.Validation_path, self.volume_size)
        self.val_volumes_all = []
        for idx in range(len(val_dataset)):
            self.val_volumes_all.append(val_dataset[idx])
        self.val_volumes_all = torch.stack(self.val_volumes_all).to(self.device)

        log_resolution = int(math.log2(self.volume_size))

        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        self.n_gen_blocks = self.generator.n_blocks
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)

        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate,
            betas=self.adam_betas
        )

        if os.path.exists(self.resume_checkpoint_path):
            self.start_step = self.load_checkpoint()
        else:
            self.start_step = 0

    def load_checkpoint(self):
        """ Loads model and optimizer states from a checkpoint file. """
        checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.mapping_network.load_state_dict(checkpoint['mapping_network'])
        self.generator_optimizer.load_state_dict(checkpoint['g_optim'])
        self.discriminator_optimizer.load_state_dict(checkpoint['d_optim'])
        self.mapping_network_optimizer.load_state_dict(checkpoint['m_optim'])
        start_step = checkpoint['step']
        print(f"Loaded checkpoint from step {start_step}. Resuming training...")
        return start_step

    def get_w(self, batch_size: int):
        """ Generates style vectors w for the generator, with optional style mixing. """
        if torch.rand(()).item() < self.style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)

            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)

            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)
        else:
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            w = self.mapping_network(z)
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)

    def get_3d_noise(self, batch_size: int):
        """ Generates a list of 3D noise tensors for each generator block. """
        noise = []
        resolution = 4

        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, resolution, device=self.device)
            n2 = torch.randn(batch_size, 1, resolution, resolution, resolution, device=self.device)

            noise.append((n1, n2))
            resolution *= 2

        return noise

    def generate_volumes(self, batch_size: int):
        """ Generates a batch of volumes using the generator. """
        w = self.get_w(batch_size)
        noise = self.get_3d_noise(batch_size)
        volumes = self.generator(w, noise)
        return volumes, w

    def save_losses(self, step, disc_loss, gen_loss, gp=None,
                    d_score_fake=None, d_score_real=None, d_score_val=None):
        """ Appends training losses and scores to the CSV file. """
        file_exists = os.path.isfile(loss_csv_path) and os.path.getsize(loss_csv_path) > 0

        with open(loss_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ['step', 'discriminator_loss', 'generator_loss', 'gradient_penalty',
                     'd_score_fake', 'd_score_real', 'd_score_val'])
            writer.writerow([
                step,
                disc_loss.item(),
                gen_loss.item(),
                gp.item() if gp is not None else '',
                d_score_fake.item() if hasattr(d_score_fake, 'item') else d_score_fake if d_score_fake is not None else '',
                d_score_real.item() if hasattr(d_score_real, 'item') else d_score_real if d_score_real is not None else '',
                d_score_val.item() if hasattr(d_score_val, 'item') else d_score_val if d_score_val is not None else ''
            ])

    def save_images(self, step, generated_volumes, real_volumes):
        """ Saves a middle slice comparison image of generated vs real volumes. """
        z_mid = self.volume_size // 2
        gen_slice = generated_volumes[0, 0, :, :, z_mid].detach().cpu()
        real_slice = real_volumes[0, 0, :, :, z_mid].detach().cpu()

        gen_slice = (gen_slice + 1) / 2
        real_slice = (real_slice + 1) / 2

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(gen_slice, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Generated Slice (z={z_mid})')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(real_slice, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Real Slice (z={z_mid})')
        plt.axis('off')

        plt.tight_layout()
        slice_path = os.path.join('results', 'slices', f'slice_step_{step}.png')
        plt.savefig(slice_path, dpi=300)
        plt.close()

    def save_checkpoint(self, step: int):
        """ Saves a training checkpoint. """
        checkpoint = {
            'step': step,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'mapping_network': self.mapping_network.state_dict(),
            'g_optim': self.generator_optimizer.state_dict(),
            'd_optim': self.discriminator_optimizer.state_dict(),
            'm_optim': self.mapping_network_optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join('results', 'checkpoints', f'ckpt_step_{step}.pth'))

    def step(self, idx: int):
        """ Performs a single training step (discriminator + generator update). """
        current_gp = None

        self.discriminator_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            generated_volumes, _ = self.generate_volumes(self.batch_size)
            fake_output = self.discriminator(generated_volumes.detach())

            real_volumes = next(self.loader).to(self.device)
            if idx % self.lazy_gradient_penalty_interval == 0:
                real_volumes.requires_grad_()
            real_output = self.discriminator(real_volumes)

            real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
            disc_loss = real_loss + fake_loss

            gp = None
            if idx % self.lazy_gradient_penalty_interval == 0:
                gp = self.gradient_penalty(real_volumes, real_output)
                current_gp = gp
                disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

            disc_loss.backward()

        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            generated_volumes, w = self.generate_volumes(self.batch_size)
            fake_output = self.discriminator(generated_volumes)

            gen_loss = self.generator_loss(fake_output)

            gen_loss.backward()

        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()

        if idx % self.log_generated_interval == 0:
            self.discriminator.eval()
            with torch.no_grad():
                generated_volumes_for_score, _ = self.generate_volumes(self.batch_size)
                d_score_fake = self.discriminator(generated_volumes_for_score).mean()

                real_volumes_for_score = next(self.loader).to(self.device)
                d_score_real = self.discriminator(real_volumes_for_score).mean()

                val_scores = []
                val_batch_size = self.batch_size
                for i in range(0, len(self.val_volumes_all), val_batch_size):
                    val_batch = self.val_volumes_all[i:i+val_batch_size]
                    val_score = self.discriminator(val_batch).mean()
                    val_scores.append(val_score)
                d_score_val = torch.stack(val_scores).mean()

            self.discriminator.train()

            print(
                f"Step {idx}: Discriminator Loss = {disc_loss.item():.4f}, Generator Loss = {gen_loss.item():.4f}, "
                f"D Fake Score = {d_score_fake.item():.4f}, D Real Score = {d_score_real.item():.4f}, D Val Score = {d_score_val.item():.4f}")
            self.save_losses(idx, disc_loss, gen_loss, current_gp,
                             d_score_fake=d_score_fake, d_score_real=d_score_real, d_score_val=d_score_val)
            self.save_images(idx, generated_volumes, real_volumes)

        if idx % self.save_checkpoint_interval == 0:
            self.save_checkpoint(idx)

    def train(self):
        """ Main training loop. """
        for i in range(self.start_step, self.start_step + self.training_steps):
            start_time = time.time()
            self.step(i)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                time_per_step = elapsed / 100
                print(f"{i + 1}/{self.start_step + self.training_steps} | "
                      f" {time_per_step:.4f} s/step | "
                      )


def main():
    """ Entry point for training. """
    configs = Configs()
    configs.init()
    configs.train()


if __name__ == '__main__':
    main()