import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatureMapping(nn.Module):
    def __init__(self, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(0, num_frequencies).float() * np.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x] if self.include_input else []
        for freq in self.freq_bands.to(x.device):
            outputs.append(torch.sin(freq * x))
            outputs.append(torch.cos(freq * x))
        return torch.cat(outputs, dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SERSDecomposition(nn.Module):
    """Generic version of the archived model that supports 1+ backgrounds."""

    def __init__(self, input_x_dim: int, input_s_dim: int = 1, hidden_dim: int = 256, z_dim: int = 128, num_backgrounds: int = 1):
        super().__init__()
        self.f_input = nn.Sequential(
            nn.Linear(input_x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.f_blocks = nn.Sequential(ResBlock(hidden_dim), ResBlock(hidden_dim))
        self.f_output = nn.Sequential(nn.Linear(hidden_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, 1))

        self.c_input = nn.Sequential(nn.Linear(input_s_dim, hidden_dim), nn.ReLU())
        self.c_blocks = nn.Sequential(ResBlock(hidden_dim), ResBlock(hidden_dim), ResBlock(hidden_dim))
        self.c_output = nn.Sequential(nn.Linear(hidden_dim, 1 + num_backgrounds), nn.Softplus())

    def forward(self, x_embed: torch.Tensor, s: torch.Tensor):
        fx = self.f_blocks(self.f_input(x_embed))
        f_x = self.f_output(fx)
        coeffs = self.c_output(self.c_blocks(self.c_input(s)))
        return coeffs, f_x


def weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def flatness_penalty(f_x: torch.Tensor, threshold: float, region_length: int = 5) -> torch.Tensor:
    f_x = f_x.view(-1)
    sq_diffs = ((f_x[1:] - f_x[:-1]) ** 2).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, region_length, device=f_x.device) / region_length
    avg_sq = F.conv1d(sq_diffs, kernel, padding=region_length // 2).squeeze()
    return torch.mean(torch.clamp(threshold - avg_sq, min=0.0))


def custom_loss(
    y_true: torch.Tensor,
    coeffs: torch.Tensor,
    f_x: torch.Tensor,
    backgrounds: list[torch.Tensor],
    scale_mode: float,
    lambda_penalty: float,
    lambda_flat: float,
    apply_flatness: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = 1e-3 * (scale_mode ** 2)
    recon = coeffs[:, 0:1] * f_x
    for bg_idx, background in enumerate(backgrounds, start=1):
        recon = recon + coeffs[:, bg_idx : bg_idx + 1] * background
    mse_loss = torch.mean((y_true - recon) ** 2)
    neg_penalty = torch.mean(torch.clamp(-f_x, min=0.0)) * scale_mode
    flat_pen = flatness_penalty(f_x, threshold=threshold) if apply_flatness else torch.tensor(0.0, device=f_x.device)
    total_loss = mse_loss + lambda_penalty * neg_penalty + lambda_flat * flat_pen
    return total_loss, recon
