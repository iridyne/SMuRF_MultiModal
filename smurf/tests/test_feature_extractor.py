# SMuRF_MultiModal/smurf/tests/test_feature_extractor.py
"""
Smoke tests for the FeatureExtractor wrapper.

These tests are intentionally lightweight and use randomly-initialized tensors.
They verify:
- the FeatureExtractor factory can be created,
- extraction methods run without error on CPU,
- output shapes match the expected (batch, out_channels) form.

Run with pytest:
    pytest -q SMuRF_MultiModal/smurf/tests/test_feature_extractor.py
"""
import sys

import pytest

try:
    import torch
except Exception:
    torch = None  # will be handled by pytest.skip below


@pytest.mark.smoke
def test_feature_extractor_basic_smoke():
    if torch is None:
        pytest.skip("torch is not available in this environment")

    # Import the wrapper from the package
    from smurf.feature_extractor import FeatureExtractor

    # Create a small default extractor for smoke testing.
    # We pick modest sizes to limit memory and execution time.
    out_channels = 8
    feature_size = 6

    extractor = FeatureExtractor.create_default(out_channels=out_channels, feature_size=feature_size)

    # Run on CPU to keep environment-agnostic
    extractor.to(torch.device("cpu"))
    extractor.eval()

    # Prepare random test inputs
    # Radiology tensor: [B, C, D, H, W]
    # Use sizes consistent with the Swin patch/window choices used in the default factory.
    x_r = torch.randn(1, 4, 32, 64, 64, dtype=torch.float32)

    # Pathology tiles: [B_tiles, C, H, W] where B_tiles = batch_size * n_tiles
    # Default pathology in_channels is 192 in the project; for the smoke test we follow that.
    tiles = torch.randn(1 * 8, 192, 16, 16, dtype=torch.float32)  # 8 tiles for a single sample

    # Run extraction (should not raise)
    with torch.no_grad():
        feats_r = extractor.extract_radiology(x_r)
        feats_p = extractor.extract_pathology(tiles, batch_size=1)

    # Basic assertions: shapes and types
    assert isinstance(feats_r, torch.Tensor), "Radiology features must be a torch.Tensor"
    assert isinstance(feats_p, torch.Tensor), "Pathology features must be a torch.Tensor"

    assert feats_r.ndim == 2, f"Radiology features must be 2D [B, out_channels], got shape {feats_r.shape}"
    assert feats_p.ndim == 2, f"Pathology features must be 2D [B, out_channels], got shape {feats_p.shape}"

    assert feats_r.shape[0] == 1, f"Radiology batch dimension should be 1 but is {feats_r.shape[0]}"
    assert feats_p.shape[0] == 1, f"Pathology batch dimension should be 1 but is {feats_p.shape[0]}"

    assert feats_r.shape[1] == out_channels, f"Radiology output channels expected {out_channels}, got {feats_r.shape[1]}"
    assert feats_p.shape[1] == out_channels, f"Pathology output channels expected {out_channels}, got {feats_p.shape[1]}"

    # Sanity: ensure parameter counting works and is > 0
    n_params = extractor.num_parameters()
    assert isinstance(n_params, int) and n_params > 0, "Expected extractor to report a positive number of parameters"

    # If we've reached here, print a brief success note (useful when running manually)
    print(f"Smoke test passed: radiology {tuple(feats_r.shape)}, pathology {tuple(feats_p.shape)}, params={n_params}")
