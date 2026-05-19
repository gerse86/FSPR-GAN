import os
import math
import time
from typing import Tuple
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from augment import AugmentPipe3D
from network import Discriminator, Generator, MappingNetwork
from utils import cycle_dataloader, PorosityDataset, DiscriminatorLoss, GeneratorLoss, GradientPenalty, TransitionDistanceLoss
from MetricChecker import MetricChecker

os.makedirs('results', exist_ok=True)
os.makedirs('results/slices', exist_ok=True)
os.makedirs('results/checkpoints', exist_ok=True)

class Configs:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.discriminator = None
        self.generator = None
        self.mapping_network = None

        self.augment_pipe = None

        self.discriminator_loss = None
        self.generator_loss = None

        self.gradient_penalty = GradientPenalty()
        self.gradient_penalty_coefficient: float = 10.

        self.Transition_loss = TransitionDistanceLoss()
        self.Transition_initial_weight = 1.0
        self.Transition_max_weight = 10.0
        self.Transition_start_step = 5000
        self.Transition_rampup_steps = 10000

        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.mapping_network_optimizer = None
        self.loader = None

        self.batch_size: int = 40
        self.d_latent: int = 128
        self.volume_size: int = 128
        self.mapping_network_layers: int = 4
        self.learning_rate: float = 1e-3
        self.mapping_network_learning_rate: float = 1e-5
        self.discriminator_repeats: int = 2
        self.gradient_accumulate_steps: int = 1
        self.adam_betas: Tuple[float, float] = (0, 0.99)
        self.style_mixing_prob: float = 0
        self.training_steps: int = 500_000
        self.n_gen_blocks: int = 0

        self.ada_target: float = 0.6
        self.ada_adjust_interval: int = 5
        self.ada_adjust_rate: float = 0.03

        self.disc_input_noise_stage1_end = 10000
        self.disc_input_noise_stage2_end = 20000
        self.disc_input_noise_sigma1 = 0.1
        self.disc_input_noise_sigma3 = 0.01

        self.lazy_gradient_penalty_interval: int = 40

        self.log_generated_interval: int = 50
        self.save_checkpoint_interval: int = 50

        self.metric_check_start_step: int = 18000
        self.metric_check_interval: int = 50
        self.metric_check_sample_num: int = 300

        self.dataset_path: str = '../DATA/Castlegate_128(300)'
        self.val_dataset_path: str = ''

        self.resume_checkpoint_path = os.path.join('results', 'checkpoints', 'ckpt_step_200.pth')

        self.metric_checker = None


    def load_checkpoint(self):
        checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.mapping_network.load_state_dict(checkpoint['mapping_network'])
        self.generator_optimizer.load_state_dict(checkpoint['g_optim'])
        self.discriminator_optimizer.load_state_dict(checkpoint['d_optim'])
        self.mapping_network_optimizer.load_state_dict(checkpoint['m_optim'])

        if 'augment_pipe_state' in checkpoint:
            self.augment_pipe.load_state_dict(checkpoint['augment_pipe_state'])

        start_step = checkpoint['step']
        print(f"Loaded checkpoint from step {start_step}. Resuming training...")
        return start_step


    def get_w(self, batch_size: int):
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

        w = self.get_w(batch_size)
        noise = self.get_3d_noise(batch_size)
        volumes = self.generator(w, noise)

        return volumes, w


    def get_disc_input_noise_sigma(self, step: int) -> float:
        if step <= self.disc_input_noise_stage1_end:
            return self.disc_input_noise_sigma1

        elif self.disc_input_noise_stage1_end < step <= self.disc_input_noise_stage2_end:
            decay_ratio = (step - self.disc_input_noise_stage1_end) / (self.disc_input_noise_stage2_end - self.disc_input_noise_stage1_end)
            return self.disc_input_noise_sigma1 - decay_ratio * (self.disc_input_noise_sigma1 - self.disc_input_noise_sigma3)

        else:
            return self.disc_input_noise_sigma3


    def save_images(self, step, generated_volumes, real_volumes):

        z_mid = self.volume_size // 2
        real_slices = real_volumes[:, 0, z_mid, :, :].detach().cpu()
        gen_slices = generated_volumes[:, 0, z_mid, :, :].detach().cpu()
        gen_slices = (gen_slices + 1) / 2  # [-1, 1] -> [0, 1]

        plt.figure(figsize=(15, 10))

        max_gen_samples = min(6, len(gen_slices))
        max_real_samples = min(3, len(real_slices))

        for i in range(6):
            plt.subplot(3, 3, i + 1)
            if i < max_gen_samples:
                plt.imshow(gen_slices[i], cmap='gray', vmin=0, vmax=1)
                plt.title(f'Generated Slice {i + 1} (z={z_mid})')
            else:
                plt.axis('off')
                plt.title('')

        for i in range(3):
            plt.subplot(3, 3, i + 7)
            if i < max_real_samples:
                plt.imshow(real_slices[i], cmap='gray', vmin=0, vmax=1)
                plt.title(f'Real Slice {i + 1} (z={z_mid})')
            else:
                plt.axis('off')
                plt.title('')

        plt.tight_layout()
        slice_path = os.path.join('results', 'slices', f'slice_step_{step}.png')
        plt.savefig(slice_path, dpi=300)
        plt.close()


    def save_checkpoint(self, step: int):

        checkpoint = {
            'step': step,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'mapping_network': self.mapping_network.state_dict(),
            'g_optim': self.generator_optimizer.state_dict(),
            'd_optim': self.discriminator_optimizer.state_dict(),
            'm_optim': self.mapping_network_optimizer.state_dict(),
            'augment_pipe_state': self.augment_pipe.state_dict(),
        }
        torch.save(checkpoint, os.path.join('results', 'checkpoints', f'ckpt_step_{step}.pth'))


    def init(self):
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

        log_resolution = int(math.log2(self.volume_size))   # 6

        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        self.n_gen_blocks = self.generator.n_blocks
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)

        self.augment_pipe = AugmentPipe3D(
            # 1
            xflip=0.6,
            yflip=0.6,
            zflip=0.6,
            rotate90=0.6,
            # 2
            erode_prob=0.3,
            dilate_prob=0.3,
            morph_kernel_size=3,
            # 3
            numeric_perturb_prob=0.5,
            sigma_boundary=0.09,
            sigma_interior=0.04
        ).to(self.device)

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

        self.metric_checker = MetricChecker(
            real_dir=self.dataset_path,
            shape=(self.volume_size, self.volume_size, self.volume_size),
            results_dir='results'
        )

        start_step = 0
        if os.path.exists(self.resume_checkpoint_path):
            start_step = self.load_checkpoint()

        return start_step


    def step(self, idx: int):

        current_ada_signal = None
        current_augment_p = self.augment_pipe.p.item()

        ############################################ Train the discriminator ############################################
        for d_repeat in range(self.discriminator_repeats):
            self.discriminator_optimizer.zero_grad()

            for i in range(self.gradient_accumulate_steps):

                generated_volumes, _ = self.generate_volumes(self.batch_size)
                augmented_fake = self.augment_pipe(generated_volumes.detach())

                real_volumes = next(self.loader).to(self.device)
                augmented_real = self.augment_pipe(real_volumes)

                current_step = idx + 1
                sigma = self.get_disc_input_noise_sigma(current_step)
                if sigma > 0:
                    fake_noise = torch.randn_like(augmented_fake) * sigma
                    augmented_fake = augmented_fake + fake_noise
                    real_noise = torch.randn_like(augmented_real) * sigma
                    augmented_real = augmented_real + real_noise

                if (idx + 1) % self.lazy_gradient_penalty_interval == 0 and d_repeat == self.discriminator_repeats - 1:
                    augmented_real.requires_grad_()

                fake_output = self.discriminator(augmented_fake)
                real_output = self.discriminator(augmented_real)

                self.augment_pipe.update_ada_stats(real_output)

                real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
                disc_loss = real_loss + fake_loss

                gp = None
                if (idx + 1) % self.lazy_gradient_penalty_interval == 0 and d_repeat == self.discriminator_repeats - 1:
                    gp = self.gradient_penalty(augmented_real, real_output)
                    current_gp = gp
                    disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

                disc_loss.backward()

            self.discriminator_optimizer.step()

        ############################################# Training the generator ############################################
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):

            generated_volumes, w = self.generate_volumes(self.batch_size)
            augmented_generated = self.augment_pipe(generated_volumes)
            fake_output = self.discriminator(augmented_generated)

            gen_loss = self.generator_loss(fake_output)

            if idx >= self.Transition_start_step:
                current_transition_penalty = self.Transition_loss(generated_volumes)

                if idx <= self.Transition_start_step + self.Transition_rampup_steps:
                    rampup_factor = (idx - self.Transition_start_step) / self.Transition_rampup_steps
                    current_transition_weight = self.Transition_initial_weight + rampup_factor * (
                                self.Transition_max_weight - self.Transition_initial_weight)
                else:
                    current_transition_weight = self.Transition_max_weight

                gen_loss += current_transition_penalty * current_transition_weight

            gen_loss.backward()

        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()

        if (idx + 1) % self.ada_adjust_interval == 0:
            current_ada_signal, current_augment_p = self.augment_pipe.adjust_p(
                target=self.ada_target,
                rate=self.ada_adjust_rate
            )

        if (idx + 1) % self.log_generated_interval == 0:

            ada_signal_str = f"{current_ada_signal:.4f}" if current_ada_signal is not None else "N/A"
            augment_p_str = f"{current_augment_p:.4f}" if current_augment_p is not None else "0.00"

            print(
                f"Step {idx + 1}: Discriminator Loss = {disc_loss.item():.4f}, Generator Loss = {gen_loss.item():.4f}, "
                f"Ada Signal = {ada_signal_str}, Augment P = {augment_p_str}")

            self.save_images(idx + 1, generated_volumes, real_volumes)

        if (idx + 1) % self.save_checkpoint_interval == 0:
            self.save_checkpoint(idx + 1)

        current_step = idx + 1
        if (current_step >= self.metric_check_start_step and
                (current_step - self.metric_check_start_step) % self.metric_check_interval == 0):
            self.metric_checker.check(
                generator=self,
                step=current_step,
                num_samples=self.metric_check_sample_num
            )

    def train(self):
        start_step = self.init()

        for i in range(start_step, self.training_steps):
            start_time = time.time()
            self.step(i)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                time_consuming = elapsed / 60
                print(f"{i + 1}/{self.training_steps} | "
                      f"{time_consuming:.2f} min/step"
                      )

if __name__ == "__main__":
    config = Configs()
    config.train()