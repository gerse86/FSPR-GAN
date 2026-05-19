import numpy as np
from pathlib import Path
import torch
import torch.utils.data
from torch import nn

def cycle_dataloader(data_loader):
    """
    Infinite loader that recycles the data loader after each epoch
    """
    while True:
        for batch in data_loader:
            yield batch

class PorosityDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, volume_size: int = 128):
        super().__init__()
        self.volume_size = volume_size
        self.paths = list(Path(path).glob('*.raw'))

        if not self.paths:
            raise ValueError(f"No .raw files were found in the path {path}！")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self.paths[index]

        if 'fid_samples' in str(file_path):
            dtype = np.float32
        else:
            dtype = np.uint8

        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)

        volume = data.reshape(1, self.volume_size, self.volume_size, self.volume_size)
        volume = torch.from_numpy(volume).float()

        if dtype == np.uint8:
            volume = (volume - 0.5) / 0.5

        return volume


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_output, fake_output):
        # -log(sigmoid(real_output))
        real_loss = -torch.log(torch.sigmoid(real_output) + 1e-8).mean()
        # -log(1 - sigmoid(fake_output))
        fake_loss = -torch.log(1 - torch.sigmoid(fake_output) + 1e-8).mean()

        return real_loss, fake_loss


class GeneratorLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, fake_output):
        # -log(sigmoid(fake_output))
        gen_loss = -torch.log(torch.sigmoid(fake_output) + 1e-8).mean()

        return gen_loss


class TransitionDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, generated_volumes):
        transition_penalty = torch.mean(1 - generated_volumes ** 2)

        return transition_penalty


class GradientPenalty(nn.Module):
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        batch_size = x.shape[0]
        gradients, *_ = torch.autograd.grad(outputs=d, inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)

        return torch.mean(norm ** 2)

