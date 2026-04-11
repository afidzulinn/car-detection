import torch
import torch.nn as nn

class CarComponentCNN(nn.Module):
    """
    Custom CNN from scratch — NO transfer learning.
    Multi-label binary classifier: 5 outputs (FL, FR, RL, RR, Hood)
    """
    def __init__(self, num_classes=5, input_size=224):
        super().__init__()

        def conv_block(in_ch, out_ch, dropout=0.25):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            )

        self.features = nn.Sequential(
            conv_block(3,   32,  0.1),
            conv_block(32,  64,  0.15),
            conv_block(64,  128, 0.2),
            conv_block(128, 256, 0.25),
            conv_block(256, 512, 0.25),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        se_w = self.se(feat).unsqueeze(-1).unsqueeze(-1)
        feat = feat * se_w
        feat = self.pool(feat)
        return self.classifier(feat)

    def predict(self, x, threshold=0.5):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
        return preds, probs


LABELS = [
    "front_left_door",
    "front_right_door",
    "rear_left_door",
    "rear_right_door",
    "hood"
]
