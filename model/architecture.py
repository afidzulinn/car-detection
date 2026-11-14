import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

LABELS = [
    "front_left_door",
    "front_right_door",
    "rear_left_door",
    "rear_right_door",
    "hood",
]

# Normalized bounding box of the 3D car across all 360-degree camera poses.
# Excludes the top UI button bar (y < 0.18H) and empty margins (~71.6% background reduction,
# 3.52x increase in effective 2D pixel density on the car).
ROI_BOX_NORM = (0.17, 0.19, 0.83, 0.62)


def crop_car_roi(img: Image.Image) -> Image.Image:
    """
    Crop the 3D car Region of Interest (ROI) from a full browser tab screenshot.
    Removes the top button container bar and surrounding empty canvas margins.
    """
    w, h = img.size
    if w > 300 and h > 300:
        x1 = int(ROI_BOX_NORM[0] * w)
        y1 = int(ROI_BOX_NORM[1] * h)
        x2 = int(ROI_BOX_NORM[2] * w)
        y2 = int(ROI_BOX_NORM[3] * h)
        return img.crop((x1, y1, x2, y2))
    return img


class ResidualConvBlock(nn.Module):
    """
    Strided Residual Convolutional Block built from scratch.
    Downsamples spatial resolution by 2x in the first 3x3 conv (reducing MACs by ~4x
    compared to end-of-block MaxPool) while preserving gradient flow via a 1x1 skip connection.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, dropout: float = 0.08):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class PlainConvBlock(nn.Module):
    """
    Standard plain convolutional block (Conv3x3 -> BN -> ReLU -> MaxPool2x2)
    used for the Baseline Standard CNN in ablation experiments.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.08):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module: highlights localized 2D regions (open doors/hood)
    across the feature map using channel-wise average and max projections.
    Uses unit-centered residual scaling (0.5 + sigmoid) so initial forward pass
    preserves feature variance relative to earlier multi-scale stages.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * (0.5 + attn)


class CarComponentCNN(nn.Module):
    """
    Custom CNN from scratch — NO transfer learning / NO pretrained weights.
    Multi-Scale Pose-Aware Spatial-Channel Attention Network (PA-SCAN):
      1. CoordConv Input Encoding (RGB + normalized X, Y coordinate channels = 5 ch)
      2. 5-Stage Residual Strided Conv Backbone with Multi-Scale Feature Fusion
         (fusing 14x14 high-acuity features for far-side doors with 7x7 context)
      3. CBAM Attention (Squeeze-and-Excitation Channel Attention + Spatial Attention)
      4. Hybrid Peak-Average Spatial Grid Pooling (MaxPool + AvgPool over 6x6 grid)
      5. Pose-Conditioned FiLM Gating + Per-Component Disentangled Classifiers
    """
    def __init__(
        self,
        num_classes: int = 5,
        num_angles: int = 8,
        input_size: int = 224,
        use_coordconv: bool = True,
        use_cbam: bool = True,
        use_film: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_angles = num_angles
        self.use_coordconv = use_coordconv
        self.use_cbam = use_cbam
        self.use_film = use_film

        in_channels = 5 if use_coordconv else 3

        # Stages 1-4: 224 -> 112 -> 56 -> 28 -> 14 (high spatial acuity for far-side doors)
        if use_coordconv:
            self.stage1_4 = nn.Sequential(
                ResidualConvBlock(in_channels, 32, stride=2, dropout=0.04),  # 224 -> 112
                ResidualConvBlock(32,  64,  stride=2, dropout=0.06),  # 112 -> 56
                ResidualConvBlock(64,  128, stride=2, dropout=0.08),  # 56  -> 28
                ResidualConvBlock(128, 256, stride=2, dropout=0.10),  # 28  -> 14
            )
            # Stage 5: 14 -> 7
            self.stage5 = ResidualConvBlock(256, 512, stride=2, dropout=0.12)
        else:
            # Baseline Standard CNN: plain Conv-BN-ReLU-MaxPool stages
            self.stage1_4 = nn.Sequential(
                PlainConvBlock(in_channels, 32, dropout=0.08),
                PlainConvBlock(32,  64,  dropout=0.10),
                PlainConvBlock(64,  128, dropout=0.12),
                PlainConvBlock(128, 256, dropout=0.15),
            )
            self.stage5 = PlainConvBlock(256, 512, dropout=0.20)

        # Channel Attention (Squeeze-and-Excitation) on Stage 5
        if use_cbam:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 512),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.se[4].weight)
            nn.init.zeros_(self.se[4].bias)
            self.spatial_attn = SpatialAttention(kernel_size=3)
            nn.init.zeros_(self.spatial_attn.conv.weight)
        else:
            self.se = None
            self.spatial_attn = None

        # Global Pose Context (1x1) & Auxiliary Camera Viewpoint Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if use_film:
            self.pose_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, num_angles),
            )
        else:
            self.pose_head = None

        # Multi-Scale Fusion (Stage 4 256ch @ 14x14 + upsampled Stage 5 512ch @ 14x14 -> 64ch)
        if use_coordconv:
            self.multiscale_proj = nn.Sequential(
                nn.Conv2d(256 + 512, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
            )
            self.grid_avg_pool = nn.AdaptiveAvgPool2d((6, 6))
            self.grid_max_pool = nn.AdaptiveMaxPool2d((6, 6))
            # 64 x 6 x 6 = 2304 spatial features
            spatial_dim = 64 * 6 * 6
        else:
            # Standard CNN baseline uses Global Average Pooling (1x1 -> 512 features)
            self.multiscale_proj = None
            self.grid_avg_pool = None
            self.grid_max_pool = None
            spatial_dim = 512

        # Pose-Conditioned FiLM Gate: modulates the 2304-dim spatial grid using camera pose
        if use_film:
            self.film_generator = nn.Sequential(
                nn.Linear(512 + num_angles, 256),
                nn.GELU(),
                nn.Linear(256, spatial_dim * 2),
            )
            # Zero-initialize final FiLM projection so (1 + tanh(gamma)) * s + beta
            # starts as an exact identity mapping at epoch 0 (Perez et al., AAAI 2018)
            nn.init.zeros_(self.film_generator[-1].weight)
            nn.init.zeros_(self.film_generator[-1].bias)
        else:
            self.film_generator = None

        # Per-Component Disentangled Heads (5 independent projections so no component
        # can cheat by reading the logits/hidden state of the other 4 components)
        self.component_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(spatial_dim, 96),
                nn.LayerNorm(96),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(96, 1),
            )
            for _ in range(num_classes)
        ])

    @staticmethod
    def _append_coords(x: torch.Tensor) -> torch.Tensor:
        """Append normalized [-1, 1] X and Y coordinate channels (CoordConv)."""
        b, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xx, yy], dim=1)

    def forward(self, x: torch.Tensor, return_pose: bool = False):
        x_in = self._append_coords(x) if self.use_coordconv else x
        f4 = self.stage1_4(x_in)                         # (B, 256, 14, 14)
        f5 = self.stage5(f4)                             # (B, 512, 7, 7)

        # Channel + Spatial Attention (CBAM) on Stage 5 (unit-centered residual scaling)
        if self.use_cbam and self.se is not None and self.spatial_attn is not None:
            se_w = 0.5 + self.se(f5).unsqueeze(-1).unsqueeze(-1)
            f5 = self.spatial_attn(f5 * se_w)

        # Global Pose Context & Auxiliary Angle Prediction
        g_feat = self.global_pool(f5).flatten(1)         # (B, 512)

        if self.use_coordconv and self.multiscale_proj is not None:
            # Fuse 14x14 Stage-4 features with upsampled Stage-5 features
            f5_up = F.interpolate(f5, size=f4.shape[-2:], mode="bilinear", align_corners=False)
            f_ms = self.multiscale_proj(torch.cat([f4, f5_up], dim=1))  # (B, 64, 14, 14)
            # Hybrid Peak + Average Pooling over 6x6 spatial grid
            s_grid = 0.5 * self.grid_avg_pool(f_ms) + 0.5 * self.grid_max_pool(f_ms)
            s_feat = s_grid.flatten(1)                   # (B, 2304)
        else:
            s_feat = g_feat                              # (B, 512)

        # Pose-Conditioned FiLM Gating
        if self.use_film and self.pose_head is not None and self.film_generator is not None:
            pose_logits = self.pose_head(g_feat)             # (B, 8)
            pose_probs = torch.softmax(pose_logits.detach(), dim=-1)  # (B, 8)
            film_params = self.film_generator(torch.cat([g_feat, pose_probs], dim=-1))
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            s_gated = (1.0 + torch.tanh(gamma)) * s_feat + beta
        else:
            pose_logits = torch.zeros(x.size(0), self.num_angles, device=x.device, dtype=x.dtype)
            s_gated = s_feat

        # Disentangled Per-Component Binary Classifiers
        logits = torch.cat([head(s_gated) for head in self.component_heads], dim=-1)

        if return_pose:
            return logits, pose_logits
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
        return preds, probs
