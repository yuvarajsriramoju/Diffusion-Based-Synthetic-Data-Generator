import torch, torch.nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler

# Tiny UNet for MNIST (28x28 grayscale)
class TinyUNet(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 3, padding=1)
        )

    def forward(self, x, t_embed):
        h = self.down(x)
        h = self.mid(h)
        return self.up(h)


class LitDDPM(pl.LightningModule):
    def __init__(self, lr=2e-4, timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.model = TinyUNet()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)

    def training_step(self, batch, _):
        x, _ = batch
        noise = torch.randn_like(x)
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (x.shape[0],), device=self.device
        ).long()
        noisy = self.noise_scheduler.add_noise(x, noise, t)
        pred = self.model(noisy, t)
        loss = torch.mean((noise - pred) ** 2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def sample(self, n=64):
        scheduler = self.noise_scheduler
        model = self.model
        model.eval()
        x = torch.randn(n, 1, 28, 28, device=self.device)
        scheduler.set_timesteps(50)  # fast sampling
        for t in scheduler.timesteps:
            pred = model(x, t)
            x = scheduler.step(pred, t, x).prev_sample
        return x.clamp(-1, 1)
