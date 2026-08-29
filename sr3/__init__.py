"""SR3 conditional diffusion model (Saharia et al., 2023).

The U-Net takes the LR condition concatenated with the noisy HR state and
predicts the noise. It is conditioned on a continuous noise level
sqrt(alpha_cumprod) rather than a discrete timestep index, which is what lets
the sampler take an arbitrary noise level at inference time.
"""
import torch
from torch import nn
from functools import partial
import numpy as np
from tqdm import tqdm
import lightning as L
from torchvision.utils import make_grid
from .unet import UNet


__all__ = ["SR3LightningModule"]


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)


class SR3LightningModule(L.LightningModule):
    def __init__(
        self,
        in_channels,
        image_size,
    ):
        """
        Initialize Gaussian Diffusion model for super-resolution.

        Args:
            in_channels (int): Number of input/output channels (e.g., 3 for RGB, 4 for RGBI)
            image_size (int): Size of the square image (height = width)
        """
        super().__init__()
        self.channels = in_channels
        self.image_size = image_size

        self.unet = UNet(
            in_channel=2 * in_channels,
            out_channel=in_channels,
            norm_groups=16,
            inner_channel=64,
            channel_mults=[1, 2, 4, 8, 16],
            attn_res=[],
            res_blocks=1,
            dropout=0,
            image_size=image_size,
        )
        weights_init_orthogonal(self.unet)

        self.loss = nn.L1Loss()
        self.set_new_noise_schedule(self.device)

    def set_new_noise_schedule(self, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = np.linspace(1e-6, 1e-2, 2000, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # Clipped, the posterior variance is 0 at the start of the chain.
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t
            - self.sqrt_recipm1_alphas_cumprod[t] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, condition_x):
        batch_size = x.shape[0]
        noise_level = (
            torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]])
            .repeat(batch_size, 1)
            .to(x.device)
        )
        x_recon = self.predict_start_from_noise(
            x,
            t=t,
            noise=self.unet(torch.cat([condition_x, x], dim=1), noise_level),
        )

        x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, condition_x):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, condition_x=condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def p_sample_intermediate(self, x, t, condition_x):
        """Mean and noise term of one reverse step, kept separate.

        Joint denoising fuses the two with different weights, so the sampler
        must not add them together here.
        """
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, condition_x=condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        variance = noise * (0.5 * model_log_variance).exp()
        return model_mean, variance

    def p_sample_loop(self, x_in):
        device = self.betas.device
        x = x_in
        shape = x.shape
        img = torch.randn(shape, device=device)
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(img, i, condition_x=x)

        return img

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise):
        return (
            continuous_sqrt_alpha_cumprod * x_start
            + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def visualize(self, sr, sr_generated, hr):
        """Grid comparing the SR input, the generated sample and the HR target."""
        sr_vis = (sr + 1) / 2
        sr_generated_vis = (sr_generated + 1) / 2
        hr_vis = (hr + 1) / 2

        comparison = []
        for i in range(sr.shape[0]):
            comparison.extend([sr_vis[i], sr_generated_vis[i], hr_vis[i]])

        grid = make_grid(comparison, nrow=3, normalize=False, padding=2)
        return grid

    def _shared_step(self, batch, batch_idx, stage):
        """Sample a full reverse chain for a few patches and log the result."""
        hr, sr, meta = batch

        # Sampling is expensive; only rank 0 runs it, on a few samples.
        if self.global_rank == 0:
            n_samples = min(8, hr.shape[0])

            hr = hr[:n_samples]
            sr = sr[:n_samples]

            sr_generated = self.p_sample_loop(sr)

            loss = self.loss(sr_generated, hr)
            self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True, batch_size=n_samples)

            grid = self.visualize(sr, sr_generated, hr)
            self.logger.experiment.add_image(
                f"{stage}/comparison", grid, self.current_epoch
            )

            return loss
        return None

    def training_step(self, batch, batch_idx):
        hr, sr, meta = batch
        bsize = hr.shape[0]
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = (
            torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t - 1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=bsize,
                )
            )
            .to(hr.device)
            .view(bsize, -1)
        )

        noise = torch.randn_like(hr)
        x_noisy = self.q_sample(
            x_start=hr,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(
                -1, 1, 1, 1
            ),
            noise=noise,
        )

        x_recon = self.unet(
            torch.cat([sr, x_noisy], dim=1), continuous_sqrt_alpha_cumprod
        )

        loss = self.loss(noise, x_recon)
        self.log("train/loss", loss, prog_bar=True, batch_size=bsize)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.unet.parameters(), lr=1e-4, weight_decay=1e-2
        )
        return optimizer
