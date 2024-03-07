import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from importlib import reload

reload(nn)


class MSIdataset(Dataset):
    """MSI dataset wrapper."""

    def __init__(self, data, xLoc, yLoc, zLoc):
        """Store dataset arrays."""
        self.len = len(data)
        self.data = data
        self.xLoc = xLoc
        self.yLoc = yLoc
        self.zLoc = zLoc

    def __getitem__(self, index):
        """Return one MSI sample."""
        return self.data[index], self.xLoc[index], self.yLoc[index], self.zLoc[index]

    def __len__(self):
        """Return sample count."""
        return self.len


class DiffusionNoiseScheduler(nn.Module):
    """Schedule Gaussian noise."""

    def __init__(self, diffusion_steps=8, beta_start=1e-4, beta_end=2e-2):
        """Build beta schedule."""
        super().__init__()
        if diffusion_steps < 1:
            raise ValueError("diffusion_steps must be at least 1")

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.diffusion_steps = diffusion_steps
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def sample_timesteps(self, batch_size, device):
        """Sample diffusion steps."""
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=device)

    def add_noise(self, latent, timesteps=None, noise=None):
        """Add Gaussian noise."""
        if timesteps is None:
            timesteps = self.sample_timesteps(latent.shape[0], latent.device)
        if noise is None:
            noise = torch.randn_like(latent)

        scale_signal = self.sqrt_alpha_bars[timesteps].unsqueeze(-1)
        scale_noise = self.sqrt_one_minus_alpha_bars[timesteps].unsqueeze(-1)
        return scale_signal * latent + scale_noise * noise, noise, timesteps


class DiffusionAutoregressiveEncoder(nn.Module):
    """Denoise via causal steps."""

    def __init__(
        self,
        d_model=256,
        n_head=8,
        num_layers=7,
        dim_feedforward=None,
        diffusion_steps=8,
        dropout=0.1,
    ):
        """Build encoder layers."""
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.noise_scheduler = DiffusionNoiseScheduler(diffusion_steps=diffusion_steps)
        self.timestep_embedding = nn.Embedding(diffusion_steps, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.autoregressive_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.denoise_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.diffusion_steps = diffusion_steps

    def _causal_mask(self, seq_len, device):
        """Build causal mask."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def _diffusion_sequence(self, latent, add_noise):
        """Build noisy step sequence."""
        timesteps = torch.arange(self.diffusion_steps, device=latent.device)
        signal_scale = self.noise_scheduler.sqrt_alpha_bars[timesteps].view(1, -1, 1)
        noise_scale = self.noise_scheduler.sqrt_one_minus_alpha_bars[timesteps].view(1, -1, 1)

        latent_seq = latent.unsqueeze(1).expand(-1, self.diffusion_steps, -1)
        if add_noise:
            noise = torch.randn_like(latent_seq)
            latent_seq = signal_scale * latent_seq + noise_scale * noise

        step_embedding = self.timestep_embedding(timesteps).unsqueeze(0)
        return latent_seq + step_embedding

    def forward(self, latent, timesteps=None, add_noise=None):
        """Encode latent spectra."""
        squeeze_batch = False
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            squeeze_batch = True
        if latent.dim() != 2:
            raise ValueError(
                "DiffusionAutoregressiveEncoder expects a 2D tensor of shape "
                "(n_spectra, d_model)"
            )

        if add_noise is None:
            add_noise = self.training

        transformer_input = self.input_norm(self._diffusion_sequence(latent, add_noise))
        mask = self._causal_mask(self.diffusion_steps, transformer_input.device)
        context = self.autoregressive_encoder(transformer_input, mask=mask)[:, -1, :]

        denoised = self.output_norm(latent + self.denoise_head(context))
        if squeeze_batch:
            denoised = denoised.squeeze(0)
        return denoised


class Model_trans(nn.Module):
    """Atnal diffusion autoencoder."""

    def __init__(
        self,
        d_mz,
        d_model=256,
        encoder_layer_num=7,
        use_decoder=False,
        decoder_layer_num=7,
        n_head=8,
        device="cpu",
        diffusion_steps=8,
        diffusion_beta_start=1e-4,
        diffusion_beta_end=2e-2,
    ):
        """Build model modules."""
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_mz, 512, device=device),
            nn.BatchNorm1d(num_features=512, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Linear(512, d_model, device=device),
            nn.BatchNorm1d(num_features=d_model, momentum=0.99, eps=1e-3),
        )

        self.transformerencoder = DiffusionAutoregressiveEncoder(
            d_model=d_model,
            n_head=n_head,
            num_layers=encoder_layer_num,
            diffusion_steps=diffusion_steps,
        )
        self.transformerencoder.noise_scheduler = DiffusionNoiseScheduler(
            diffusion_steps=diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
        )

        self.recover = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(num_features=512, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Linear(512, d_mz),
            nn.Softmax(dim=-1),
        )
        self.d_model = d_model
        self.position_Enbedding = None
        self.log = {"training loss": [], "testing loss": []}
        self.to(device)

    def encode(self, input, add_noise=None):
        """Encode MSI spectra."""
        h = self.backbone(input)
        return self.transformerencoder(h, add_noise=add_noise)

    def forward(self, input):
        """Reconstruct MSI spectra."""
        h = self.encode(input)
        return self.TIC_norm(self.recover(h))

    def TIC_norm(self, input):
        """Normalize TIC."""
        denom = torch.sum(input, dim=-1, keepdim=True).clamp_min(1e-12)
        return input / denom


def categorical_crossentropy(pred, label):
    """Categorical cross entropy."""
    pred = torch.clip(pred, min=1e-7, max=1.0 - 1e-7)
    loss = torch.sum(-label * torch.log(pred)) / pred.shape[0]
    return loss
