"""Image value model of Sec. III-D: frozen ImageNet-pretrained ResNet-34 encoder + 2-layer MLP (Eq. 7).

    v_t = MLP(ResNet34(s_t)),   psi* = argmin_psi L_MSE(MLP_psi(s_t), v_t)

Only the MLP head is trained. Checkpoints store the head and the preprocessing config; the encoder is re-created
from torchvision's ImageNet weights at load time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_reasoner.value.base import frame_of

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet34Value(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        pretrained: bool = True,
        input_size: int = 224,
        degrade_resolution: Optional[int] = None,
    ):
        """
        Args:
            hidden_dim: width of the hidden layer of the 2-layer MLP head.
            pretrained: load torchvision's ImageNet weights (set False in tests to avoid a download).
            input_size: square resolution fed to ResNet-34.
            degrade_resolution: if set, frames are first resized to this resolution (e.g. the world model's
                output resolution) and then up to `input_size`, so that real training frames look like
                world-model predictions. Off by default.
        """
        super().__init__()
        import torchvision

        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = torchvision.models.resnet34(weights=weights)
        encoder.fc = nn.Identity()
        for p in encoder.parameters():
            p.requires_grad_(False)
        self.encoder = encoder.eval()
        self.head = nn.Sequential(nn.Linear(512, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.config: Dict[str, Any] = {
            "hidden_dim": hidden_dim,
            "input_size": input_size,
            "degrade_resolution": degrade_resolution,
        }
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # frozen encoder: BatchNorm statistics are never updated
        return self

    def preprocess(self, images) -> torch.Tensor:
        """uint8 (B, H, W, 3) array/tensor -> normalized float (B, 3, S, S) on the module's device."""
        x = torch.as_tensor(np.asarray(images) if not torch.is_tensor(images) else images)
        x = x.to(self._mean.device).permute(0, 3, 1, 2).float() / 255.0
        degrade = self.config["degrade_resolution"]
        if degrade:
            x = F.interpolate(x, size=(degrade, degrade), mode="bilinear", antialias=True, align_corners=False)
        size = self.config["input_size"]
        if x.shape[-2:] != (size, size):
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        return (x - self._mean) / self._std

    @torch.no_grad()
    def encode(self, images) -> torch.Tensor:
        """Frozen ResNet-34 features, (B, 512)."""
        return self.encoder(self.preprocess(images))

    def forward(self, images) -> torch.Tensor:
        return self.head(self.encode(images)).squeeze(-1)

    @torch.no_grad()
    def predict(self, images, batch_size: int = 64) -> np.ndarray:
        self.eval()
        out = [self.forward(images[i:i + batch_size]).float().cpu() for i in range(0, len(images), batch_size)]
        return torch.cat(out).numpy()

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        torch.save({"head": self.head.state_dict(), "config": dict(self.config), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, pretrained: bool = True, device: str = "cpu") -> "ResNet34Value":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(pretrained=pretrained, **ckpt["config"])
        model.head.load_state_dict(ckpt["head"])
        return model.to(device).eval()


class ImageValueFunction:
    """Callable used by the tree search: list of world-model states -> values (Alg. 1 line 6)."""

    def __init__(self, model: ResNet34Value, batch_size: int = 64):
        self.model = model.eval()
        self.batch_size = batch_size

    def __call__(self, states: Sequence[Any]) -> np.ndarray:
        frames = np.stack([frame_of(s) for s in states]).astype(np.uint8)
        return self.model.predict(frames, self.batch_size).astype(np.float64)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "ImageValueFunction":
        return cls(ResNet34Value.load(path, device=device))
