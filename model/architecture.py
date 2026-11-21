"""
Multi-Scale PA-SCAN Architecture (v1 & v2) with Orthogonal Ablation Flags (P1-06)
and P2 Novelty Modules (P2-01, P2-02, P2-03, P2-04):

  - P2-01 (use_elevation_pose): Joint 8-class Azimuth + 3-class Elevation camera pose
    auxiliary heads and 11-D viewpoint-conditioned FiLM modulation.
  - P2-02 (use_component_attention): Pose-Conditioned Component-Specific Spatial Attention
    (5 dedicated 14x14 spatial attention maps A_1..A_5 anchored to 2D CoordConv + camera pose).
  - P2-03 (use_visibility_head): Auxiliary 5-component Visibility / Observability Prediction Head
    supporting visibility-bin diagnostics and selective prediction (Coverage vs. Risk).
  - P2-04: Exposes 14x14 part attention maps A_k in `return_aux=True` for training-time
    counterfactual part-mask auxiliary supervision (L_mask) with zero inference-time oracle dependency.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from preprocess import FULL_VEHICLE_ROI_BOX_NORM, LEGACY_ROI_BOX_NORM, crop_with_policy

LABELS = [
    "front_left_door",
    "front_right_door",
    "rear_left_door",
    "rear_right_door",
    "hood",
]

ROI_BOX_NORM = LEGACY_ROI_BOX_NORM


def crop_car_roi(img: Image.Image, policy: str = "legacy_fixed") -> Image.Image:
    return crop_with_policy(img, policy=policy)


@dataclass
class ModelConfig:
    num_classes: int = 5
    num_angles: int = 8
    num_elevations: int = 3
    input_size: int = 224
    backbone: str = "residual"               # "residual" | "plain"
    use_coordconv: bool = True               # ONLY toggles 5-ch (RGB+XY) vs 3-ch (RGB)
    use_multiscale: bool = True              # ONLY toggles Stage4+Stage5 fusion vs Stage5-only
    use_cbam: bool = True                    # ONLY toggles Unit-Centered Residual CBAM
    pooling: str = "hybrid_grid"             # "hybrid_grid" (6x6) | "global_avg" (1x1)
    use_film: bool = True                    # ONLY toggles Pose-Conditioned FiLM
    detach_pose_in_film: bool = True         # P1-07: stop-gradient on pose_probs into FiLM
    use_elevation_pose: bool = False         # P2-01: add 3-class elevation auxiliary pose head
    use_component_attention: bool = False    # P2-02: add pose-conditioned per-component spatial attention
    use_visibility_head: bool = False        # P2-03: add 5-component auxiliary visibility head

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualConvBlock(nn.Module):
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


class ComponentSpatialAttention(nn.Module):
    """
    P2-02: Pose-Conditioned Component-Specific Spatial Feature Extractor.
    Appends 2D normalized coordinates (X, Y) to the 14x14 multi-scale feature map
    and modulates a 3x3 spatial attention trunk with the camera pose probabilities
    (azimuth + elevation) so each of the K=5 components generates a viewpoint-steered
    14x14 spatial attention map A_k in [0, 1]^(1 x 14 x 14).
    """

    def __init__(self, in_channels: int = 64, num_components: int = 5, pose_dim: int = 8):
        super().__init__()
        self.num_components = num_components
        self.pose_dim = pose_dim
        self.spatial_trunk = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.pose_to_query = nn.Sequential(
            nn.Linear(pose_dim, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )
        nn.init.zeros_(self.pose_to_query[-1].weight)
        nn.init.zeros_(self.pose_to_query[-1].bias)

        self.part_heads = nn.Conv2d(in_channels, num_components, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_map: torch.Tensor, pose_vec: Optional[torch.Tensor] = None):
        b, c, h, w = f_map.shape
        device, dtype = f_map.device, f_map.dtype
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)

        h_map = self.spatial_trunk(torch.cat([f_map, xx, yy], dim=1))
        if pose_vec is not None and pose_vec.size(-1) == self.pose_dim:
            p_gate = 1.0 + torch.tanh(self.pose_to_query(pose_vec)).unsqueeze(-1).unsqueeze(-1)
            h_map = h_map * p_gate

        attn_maps = self.sigmoid(self.part_heads(h_map))  # (B, K, H, W)
        pooled_per_part = []
        for k in range(self.num_components):
            ak = attn_maps[:, k : k + 1, :, :]  # (B, 1, H, W)
            weighted = f_map * (0.5 + ak)
            norm_avg = (f_map * ak).sum(dim=(-2, -1)) / (ak.sum(dim=(-2, -1)) + 1e-6)
            norm_max = F.adaptive_max_pool2d(weighted, 1).flatten(1)
            pooled_per_part.append(0.5 * norm_avg + 0.5 * norm_max)  # (B, C)
        return pooled_per_part, attn_maps


class CarComponentCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        num_angles: int = 8,
        input_size: int = 224,
        use_coordconv: bool = True,
        use_cbam: bool = True,
        use_film: bool = True,
        backbone: Optional[str] = None,
        use_multiscale: Optional[bool] = None,
        pooling: Optional[str] = None,
        num_elevations: int = 3,
        use_elevation_pose: bool = False,
        use_component_attention: bool = False,
        use_visibility_head: bool = False,
        detach_pose_in_film: bool = True,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        if config is not None:
            num_classes = config.num_classes
            num_angles = config.num_angles
            num_elevations = config.num_elevations
            input_size = config.input_size
            use_coordconv = config.use_coordconv
            use_cbam = config.use_cbam
            use_film = config.use_film
            backbone = config.backbone
            use_multiscale = config.use_multiscale
            pooling = config.pooling
            use_elevation_pose = config.use_elevation_pose
            use_component_attention = config.use_component_attention
            use_visibility_head = config.use_visibility_head
            detach_pose_in_film = config.detach_pose_in_film

        self.backbone_type = backbone if backbone is not None else "residual"
        self.use_multiscale = True if use_multiscale is None else bool(use_multiscale)
        self.pooling_type = pooling if pooling is not None else "hybrid_grid"

        self.num_classes = num_classes
        self.num_angles = num_angles
        self.num_elevations = num_elevations
        self.use_coordconv = bool(use_coordconv)
        self.use_cbam = bool(use_cbam)
        self.use_film = bool(use_film)
        self.use_elevation_pose = bool(use_elevation_pose)
        self.use_component_attention = bool(use_component_attention)
        self.use_visibility_head = bool(use_visibility_head)
        self.detach_pose_in_film = bool(detach_pose_in_film)

        in_channels = 5 if self.use_coordconv else 3

        if self.backbone_type == "residual":
            self.stage1_4 = nn.Sequential(
                ResidualConvBlock(in_channels, 32, stride=2, dropout=0.04),
                ResidualConvBlock(32, 64, stride=2, dropout=0.06),
                ResidualConvBlock(64, 128, stride=2, dropout=0.08),
                ResidualConvBlock(128, 256, stride=2, dropout=0.10),
            )
            self.stage5 = ResidualConvBlock(256, 512, stride=2, dropout=0.12)
        elif self.backbone_type == "plain":
            self.stage1_4 = nn.Sequential(
                PlainConvBlock(in_channels, 32, dropout=0.08),
                PlainConvBlock(32, 64, dropout=0.10),
                PlainConvBlock(64, 128, dropout=0.12),
                PlainConvBlock(128, 256, dropout=0.15),
            )
            self.stage5 = PlainConvBlock(256, 512, dropout=0.20)
        else:
            raise ValueError(f"Unknown backbone '{self.backbone_type}'.")

        if self.use_cbam:
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

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        pose_cond_dim = num_angles + (num_elevations if self.use_elevation_pose else 0)

        if self.use_film or self.use_elevation_pose or self.use_component_attention:
            self.pose_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, num_angles),
            )
            if self.use_elevation_pose:
                self.elevation_head = nn.Sequential(
                    nn.Linear(512, 64),
                    nn.GELU(),
                    nn.Linear(64, num_elevations),
                )
            else:
                self.elevation_head = None
        else:
            self.pose_head = None
            self.elevation_head = None

        if self.pooling_type == "hybrid_grid":
            proj_in_ch = (256 + 512) if self.use_multiscale else 512
            self.multiscale_proj = nn.Sequential(
                nn.Conv2d(proj_in_ch, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
            )
            self.grid_avg_pool = nn.AdaptiveAvgPool2d((6, 6))
            self.grid_max_pool = nn.AdaptiveMaxPool2d((6, 6))
            spatial_dim = 64 * 6 * 6
            part_feat_ch = 64
        elif self.pooling_type == "global_avg":
            self.multiscale_proj = None
            self.grid_avg_pool = None
            self.grid_max_pool = None
            spatial_dim = 512
            part_feat_ch = 512
        else:
            raise ValueError(f"Unknown pooling '{self.pooling_type}'.")

        if self.use_component_attention and self.multiscale_proj is not None:
            self.component_spatial_attn = ComponentSpatialAttention(
                in_channels=part_feat_ch,
                num_components=num_classes,
                pose_dim=pose_cond_dim,
            )
            head_in_dim = spatial_dim + part_feat_ch
        else:
            self.component_spatial_attn = None
            head_in_dim = spatial_dim

        if self.use_film:
            self.film_generator = nn.Sequential(
                nn.Linear(512 + pose_cond_dim, 256),
                nn.GELU(),
                nn.Linear(256, spatial_dim * 2),
            )
            nn.init.zeros_(self.film_generator[-1].weight)
            nn.init.zeros_(self.film_generator[-1].bias)
        else:
            self.film_generator = None

        if self.use_visibility_head:
            self.visibility_head = nn.Sequential(
                nn.Linear(512 + pose_cond_dim, 64),
                nn.GELU(),
                nn.Linear(64, num_classes),
            )
        else:
            self.visibility_head = None

        self.component_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in_dim, 96),
                nn.LayerNorm(96),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(96, 1),
            )
            for _ in range(num_classes)
        ])

    @staticmethod
    def _append_coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xx, yy], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_pose: bool = False,
        return_aux: bool = False,
        oracle_pose_probs: Optional[torch.Tensor] = None,
        oracle_elev_probs: Optional[torch.Tensor] = None,
    ):
        x_in = self._append_coords(x) if self.use_coordconv else x
        f4 = self.stage1_4(x_in)  # (B, 256, 14, 14)
        f5 = self.stage5(f4)      # (B, 512, 7, 7)

        if self.use_cbam and self.se is not None and self.spatial_attn is not None:
            se_w = 0.5 + self.se(f5).unsqueeze(-1).unsqueeze(-1)
            f5 = self.spatial_attn(f5 * se_w)

        g_feat = self.global_pool(f5).flatten(1)  # (B, 512)

        # Compute pose distributions first so both FiLM and ComponentSpatialAttention can use them
        elev_logits = None
        if self.pose_head is not None:
            pose_logits = self.pose_head(g_feat)  # (B, 8)
            p_in = pose_logits.detach() if self.detach_pose_in_film else pose_logits
            pose_probs = oracle_pose_probs if oracle_pose_probs is not None else torch.softmax(p_in, dim=-1)
            if self.use_elevation_pose and self.elevation_head is not None:
                elev_logits = self.elevation_head(g_feat)  # (B, 3)
                e_in = elev_logits.detach() if self.detach_pose_in_film else elev_logits
                elev_probs = oracle_elev_probs if oracle_elev_probs is not None else torch.softmax(e_in, dim=-1)
                pose_cond_vec = torch.cat([pose_probs, elev_probs], dim=-1)
            else:
                pose_cond_vec = pose_probs
        else:
            pose_logits = torch.zeros(x.size(0), self.num_angles, device=x.device, dtype=x.dtype)
            pose_cond_vec = torch.zeros(x.size(0), self.num_angles, device=x.device, dtype=x.dtype)

        part_features = None
        part_attn_maps = None
        f_ms = None

        if self.pooling_type == "hybrid_grid" and self.multiscale_proj is not None:
            if self.use_multiscale:
                f5_up = F.interpolate(f5, size=f4.shape[-2:], mode="bilinear", align_corners=False)
                f_ms = self.multiscale_proj(torch.cat([f4, f5_up], dim=1))  # (B, 64, 14, 14)
            else:
                f_ms = self.multiscale_proj(f5)  # (B, 64, 7, 7)
            s_grid = 0.5 * self.grid_avg_pool(f_ms) + 0.5 * self.grid_max_pool(f_ms)
            s_feat = s_grid.flatten(1)  # (B, 2304)
            if self.component_spatial_attn is not None:
                part_features, part_attn_maps = self.component_spatial_attn(f_ms, pose_cond_vec)
        else:
            s_feat = g_feat  # (B, 512)

        if self.use_film and self.film_generator is not None:
            cond_vec = torch.cat([g_feat, pose_cond_vec], dim=-1)
            film_params = self.film_generator(cond_vec)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            s_gated = (1.0 + torch.tanh(gamma)) * s_feat + beta
        else:
            s_gated = s_feat

        vis_logits = None
        if self.use_visibility_head and self.visibility_head is not None:
            vis_logits = self.visibility_head(torch.cat([g_feat, pose_cond_vec], dim=-1))  # (B, 5)

        if part_features is not None:
            head_outputs = [
                self.component_heads[k](torch.cat([s_gated, part_features[k]], dim=-1))
                for k in range(self.num_classes)
            ]
        else:
            head_outputs = [head(s_gated) for head in self.component_heads]

        logits = torch.cat(head_outputs, dim=-1)

        if return_aux:
            return logits, {
                "pose_logits": pose_logits,
                "elevation_logits": elev_logits,
                "visibility_logits": vis_logits,
                "f4": f4,
                "f5": f5,
                "f_ms": f_ms,
                "part_attn_maps": part_attn_maps,
            }
        if return_pose:
            return logits, pose_logits
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.50):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int()
        return preds, probs
