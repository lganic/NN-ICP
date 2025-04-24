import os
import random
from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# === Dataset ===
class InpaintingDataset(Dataset):
    def __init__(self, root_folder, image_size=256):
        self.image_paths = glob(os.path.join(root_folder, "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)  # [C, H, W]

        mask = torch.zeros_like(image[:1])  # [1, H, W]
        # Random block mask (box)
        h, w = image.shape[1:]

        num_rects = random.randint(3, 7)  # Choose how many rectangles per image

        for _ in range(num_rects):
            # Small rectangles: size 1/16 to 1/8 of the image
            mh = random.randint(h // 10, h // 6)
            mw = random.randint(w // 10, w // 6)
            top = random.randint(0, h - mh)
            left = random.randint(0, w - mw)
            mask[:, top:top + mh, left:left + mw] = 1

        # Combine image + mask into 4 channels
        masked_image = image.clone()
        masked_image = masked_image * (1 - mask)
        input_tensor = torch.cat([masked_image, mask], dim=0)  # [4, H, W]

        return input_tensor, image, mask

# === Sample training loop ===
def train(model, train_dataloader, val_dataloader, optimizer, scheduler, noise_scheduler, epochs=10, device="cuda"):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for i, (x, target, mask) in enumerate(train_dataloader):
            x, target, mask = x.to(device), target.to(device), mask.to(device)

            t = torch.randint(0, noise_scheduler.timesteps, (x.shape[0],), device=device).long()
            noise = torch.randn_like(target)
            noised = noise_scheduler.q_sample(target, t, noise)

            noised = noised * mask + target * (1 - mask)
            pred = model(torch.cat([noised * (1 - mask), mask], dim=1), t)
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f"Epoch {epoch} | Step {i} | Train Loss: {loss.item():.4f}")
            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Train Loss: {loss.item():.4f}")

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, target, mask in val_dataloader:
                x, target, mask = x.to(device), target.to(device), mask.to(device)
                t = torch.randint(0, noise_scheduler.timesteps, (x.shape[0],), device=device).long()
                noise = torch.randn_like(target)
                noised = noise_scheduler.q_sample(target, t, noise)
                noised = noised * mask + target * (1 - mask)
                pred = model(torch.cat([noised * (1 - mask), mask], dim=1), t)
                val_loss += F.mse_loss(pred, noise, reduction='sum').item()

        val_loss /= len(val_dataloader.dataset)
        print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f}")


# === Noise Scheduler ===
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise):
        """Diffusion forward process."""
        a_bar = self.alpha_bars.to(x_start.device)[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x_start + torch.sqrt(1 - a_bar) * noise

# === Usage ===
if __name__ == "__main__":
    from network import UNet

    train_dataset = InpaintingDataset("/blue/aosmith1/logan.boehm/processed_texture_dataset/train")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    val_dataset = InpaintingDataset("/blue/aosmith1/logan.boehm/processed_texture_dataset/val")
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)

    model = UNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    noise_scheduler = NoiseScheduler()

    train(model, train_dataloader, val_dataloader, optimizer, scheduler, noise_scheduler, epochs=20)
