import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(channels):
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_residual=True):
        super().__init__()
        self.use_residual = use_residual and in_ch == out_ch
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            get_norm_layer(out_ch),
            nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            get_norm_layer(out_ch),
            nn.SiLU()
        )

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_emb(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.block2(h)
        return x + h if self.use_residual else h

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(x_)
        q, k, v = qkv.chunk(3, dim=1)
        attn = torch.einsum("bci,bcj->bij", q, k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v)
        out = self.out_proj(out).view(B, C, H, W)
        return x + out

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=128, time_emb_dim=512):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Channel progression
        chs = [in_channels, base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*8]

        # Downsampling path: two residual blocks per scale
        self.downs = nn.ModuleList()
        self.attn_downs = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.downs.append(nn.ModuleList([
                ResidualBlock(chs[i],   chs[i+1], time_emb_dim),
                ResidualBlock(chs[i+1], chs[i+1], time_emb_dim),
            ]))
            self.attn_downs.append(AttentionBlock(chs[i+1]) if i >= 2 else nn.Identity())

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = ResidualBlock(chs[-1], chs[-1], time_emb_dim)
        self.bot_attn = AttentionBlock(chs[-1])
        self.bot2 = ResidualBlock(chs[-1], chs[-1], time_emb_dim)

        # Upsampling path: two residual blocks per scale
        self.ups = nn.ModuleList()
        self.attn_ups = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.ups.append(nn.ModuleList([
                ResidualBlock(chs[i+1]*2, chs[i], time_emb_dim),  # Concatenate skip
                ResidualBlock(chs[i],     chs[i], time_emb_dim)
            ]))
            self.attn_ups.append(AttentionBlock(chs[i]) if i >= 2 else nn.Identity())

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        skip_connections = []

        # Down path
        for down_blocks, attn in zip(self.downs, self.attn_downs):
            for block in down_blocks:
                x = block(x, t_emb)
            x = attn(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bot1(x, t_emb)
        x = self.bot_attn(x)
        x = self.bot2(x, t_emb)

        # Up path
        for up_blocks, attn in zip(self.ups, self.attn_ups):
            skip = skip_connections.pop()
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
            for block in up_blocks:
                x = block(x, t_emb)
            x = attn(x)

        return self.final(x)

# === Test run ===
if __name__ == '__main__':
    import torch
    from torchvision.utils import save_image
    import os

    model = UNet(in_channels=4, out_channels=3)
    model.eval()

    B, H, W = 1, 256, 256
    image = torch.randn(B, 3, H, W)
    mask = torch.randint(0, 2, (B, 1, H, W))
    timestep = torch.randint(0, 1000, (B,))

    input_tensor = torch.cat([image, mask], dim=1)

    with torch.no_grad():
        output = model(input_tensor, timestep)

    output_vis = (output - output.min()) / (output.max() - output.min() + 1e-8)

    os.makedirs("output_images", exist_ok=True)
    save_image(output_vis, "output_images/unet_output.png")
    print("Saved to output_images/unet_output.png")
