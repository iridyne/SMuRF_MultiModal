"""
Feature extractor wrapper for SMuRF Swin-based models.

This module provides a small, robust wrapper around the Swin-based radiology
and pathology extractors implemented in `smurf.models`. The wrapper exposes a
consistent, easy-to-use API for extracting feature vectors from:

- 3D radiology volumes (CT): model expects input shape [B, C, D, H, W]
- 2D pathology tiles / embeddings: model expects input shape [B_tiles, C, H, W]
  and requires `batch_size` (number of original samples) to group tiles back
  into samples: B_tiles == batch_size * n_tiles_per_sample

The wrapper intentionally keeps runtime dependencies limited to `torch` and the
models defined in the package. It includes convenience helpers to move models to
device, set evaluation mode, load/save weights, and a small smoke test that can
be used locally.

Note:
- The smoke test uses randomly-initialized models/tensors and is meant for
  development sanity checks only.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from .models import (
    SwinTransformerPathologyModel,
    SwinTransformerRadiologyModel,
)


class FeatureExtractor:
    """
    Wrapper around SwinTransformerRadiologyModel and SwinTransformerPathologyModel.

    Typical usage:
        # Create with existing models
        extractor = FeatureExtractor(radiology_model=my_radiology, pathology_model=my_pathology)
        extractor.to('cuda')
        extractor.eval()
        feats_r = extractor.extract_radiology(x_r)         # [B, out_channels]
        feats_p = extractor.extract_pathology(x_tiles, B)  # [B, out_channels]

    Or create a default pair (useful for smoke tests):
        extractor = FeatureExtractor.create_default()
    """

    def __init__(
        self,
        radiology_model: Optional[nn.Module] = None,
        pathology_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        # Default device selection if not provided
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # If models are not provided, do not automatically build heavy models here.
        # Use `create_default` to instantiate standard-config models for testing.
        self.radiology = radiology_model
        self.pathology = pathology_model

        if self.radiology is not None:
            self.radiology.to(self.device)
        if self.pathology is not None:
            self.pathology.to(self.device)

    # ---------------------------
    # Factory / helpers
    # ---------------------------
    @classmethod
    def create_default(cls, *, out_channels: int = 24, feature_size: int = 12) -> "FeatureExtractor":
        """
        Create an extractor with default Swin configs that match the project's
        common instantiation choices.

        Returns a FeatureExtractor with freshly initialized models on CPU.
        """
        # Radiology defaults used in the original Model class (smaller variant)
        radiology_model = SwinTransformerRadiologyModel(
            patch_size=(1, 2, 2),
            window_size=[[4, 4, 4], [4, 4, 4]],
            in_channels=4,
            out_channels=out_channels,
            depths=(2, 2),
            num_heads=(3, 6),
            feature_size=feature_size,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3,
        )

        # Pathology defaults used in the original PModel/Model
        pathology_model = SwinTransformerPathologyModel(
            patch_size=(2, 2),
            window_size=[[4, 4], [4, 4]],
            in_channels=192,
            out_channels=out_channels,
            depths=(2, 2),
            num_heads=(3, 6),
            feature_size=feature_size,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=2,
        )

        return cls(radiology_model=radiology_model, pathology_model=pathology_model, device=torch.device("cpu"))

    def to(self, device: torch.device):
        """
        Move both models to `device`. Accepts either a `torch.device` or a string.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.radiology is not None:
            self.radiology.to(self.device)
        if self.pathology is not None:
            self.pathology.to(self.device)
        return self

    def eval(self):
        """Set both extractors to evaluation mode."""
        if self.radiology is not None:
            self.radiology.eval()
        if self.pathology is not None:
            self.pathology.eval()
        return self

    def train(self, mode: bool = True):
        """Set both extractors to training mode if mode=True else evaluation mode."""
        if self.radiology is not None:
            self.radiology.train(mode)
        if self.pathology is not None:
            self.pathology.train(mode)
        return self

    # ---------------------------
    # Loading / saving helpers
    # ---------------------------
    def load_state_dict(self, radiology_state: Optional[dict] = None, pathology_state: Optional[dict] = None, strict: bool = True):
        """
        Load state dicts into the wrapped models if provided.
        """
        if radiology_state is not None:
            if self.radiology is None:
                raise RuntimeError("Radiology model is not set.")
            self.radiology.load_state_dict(radiology_state, strict=strict)
        if pathology_state is not None:
            if self.pathology is None:
                raise RuntimeError("Pathology model is not set.")
            self.pathology.load_state_dict(pathology_state, strict=strict)

    def state_dict(self) -> Tuple[Optional[dict], Optional[dict]]:
        """Return a tuple of (radiology_state_dict, pathology_state_dict)."""
        rad_state = self.radiology.state_dict() if self.radiology is not None else None
        path_state = self.pathology.state_dict() if self.pathology is not None else None
        return rad_state, path_state

    # ---------------------------
    # Extraction methods
    # ---------------------------
    def extract_radiology(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract radiology features.

        Args:
            x: Tensor with shape [B, C, D, H, W].

        Returns:
            Tensor with shape [B, out_channels].
        """
        if self.radiology is None:
            raise RuntimeError("Radiology model is not set.")
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        # Ensure float32 for model
        if not torch.is_floating_point(x):
            x = x.float()
        with torch.no_grad():
            feats = self.radiology(x)
        return feats

    def extract_pathology(self, x_tiles: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Extract pathology features.

        Args:
            x_tiles: Tensor with shape [B_tiles, C, H, W], where B_tiles == batch_size * n_tiles_per_sample
            batch_size: the number of original samples grouped in x_tiles

        Returns:
            Tensor with shape [batch_size, out_channels]
        """
        if self.pathology is None:
            raise RuntimeError("Pathology model is not set.")
        if not isinstance(x_tiles, torch.Tensor):
            x_tiles = torch.tensor(x_tiles)
        x_tiles = x_tiles.to(self.device)
        if not torch.is_floating_point(x_tiles):
            x_tiles = x_tiles.float()
        with torch.no_grad():
            feats = self.pathology(x_tiles, batch_size)
        return feats

    def extract_multimodal(self, ct_tumor: torch.Tensor, ct_lymph: torch.Tensor, path_tiles: torch.Tensor, path_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method to extract features for all three modalities used in Model.forward.

        Returns:
            (features_tumor, features_lymph, features_pathology)
        """
        feats_tumor = self.extract_radiology(ct_tumor)
        feats_lymph = self.extract_radiology(ct_lymph)
        feats_path = self.extract_pathology(path_tiles, path_batch_size)
        return feats_tumor, feats_lymph, feats_path

    # ---------------------------
    # Utilities
    # ---------------------------
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters across both models."""
        total = 0
        for m in (self.radiology, self.pathology):
            if m is None:
                continue
            total += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total


# ---------------------------
# Simple smoke test (runs when executed directly)
# ---------------------------
def _smoke_test() -> None:
    """Basic smoke test using random tensors (does not require real data)."""
    print("Running FeatureExtractor smoke test (random tensors)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor.create_default()
    extractor.to(device)
    extractor.eval()

    # Radiology: B=1, C=4, D=32, H=64, W=64
    x_r = torch.randn(1, 4, 32, 64, 64, device=device)
    # Lymph same shape for smoke test
    x_l = torch.randn(1, 4, 32, 64, 64, device=device)
    # Pathology: one sample with 8 tiles, in_channels=192, tile 16x16
    tiles = torch.randn(1 * 8, 192, 16, 16, device=device)

    try:
        feats_r = extractor.extract_radiology(x_r)
        feats_l = extractor.extract_radiology(x_l)
        feats_p = extractor.extract_pathology(tiles, batch_size=1)
        print("Radiology features shape:", tuple(feats_r.shape))
        print("Lymph features shape:", tuple(feats_l.shape))
        print("Pathology features shape:", tuple(feats_p.shape))
    except Exception as exc:
        print("Smoke test failed with exception:", exc)
        raise


if __name__ == "__main__":
    _smoke_test()
