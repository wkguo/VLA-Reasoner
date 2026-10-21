"""Tests for the ResNet-34 value model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from vla_reasoner.value.model import ImageValueFunction, ResNet34Value  # noqa: E402
from vla_reasoner.value.train import encode_samples, fit_value_head  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ResNet34Value(hidden_dim=16, pretrained=False, input_size=64)  # weights=None: no download


def test_forward_shape_and_frozen_encoder(model):
    images = np.random.default_rng(0).integers(0, 255, size=(2, 48, 40, 3), dtype=np.uint8)
    assert model.encode(images).shape == (2, 512)
    model.train()
    assert not model.encoder.training  # BatchNorm statistics stay frozen
    out = model(images)
    assert out.shape == (2,)
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in model.head.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())


def test_degrade_resolution_preprocessing():
    m = ResNet34Value(hidden_dim=8, pretrained=False, input_size=32, degrade_resolution=8)
    x = m.preprocess(np.zeros((1, 64, 64, 3), dtype=np.uint8))
    assert x.shape == (1, 3, 32, 32)


def test_training_reduces_mse_and_roundtrip(model, tmp_path):
    rng = np.random.default_rng(1)
    samples = []
    for ep in range(6):  # images whose brightness encodes progress
        labels = np.linspace(0, 1, 8).astype(np.float32)
        frames = np.stack([np.full((32, 32, 3), int(255 * y), dtype=np.uint8) for y in labels])
        frames = np.clip(frames.astype(int) + rng.integers(-5, 5, frames.shape), 0, 255).astype(np.uint8)
        samples.append((frames, labels, ep))
    feats, labels, groups = encode_samples(model, samples, batch_size=4)
    assert feats.shape == (48, 512) and labels.shape == (48,) and len(np.unique(groups)) == 6
    hist = fit_value_head(model, feats, labels, groups, epochs=150, lr=1e-2, batch_size=16, val_fraction=0.34)
    assert hist[-1]["train_mse"] < 0.5 * hist[0]["train_mse"]
    assert "val_mse" in hist[-1]

    path = str(tmp_path / "value.pt")
    model.save(path)
    loaded = ResNet34Value.load(path, pretrained=False)
    # the encoder is re-created (not saved): copy it to compare heads on identical features
    loaded.encoder.load_state_dict(model.encoder.state_dict())
    frames = samples[0][0]
    np.testing.assert_allclose(loaded.predict(frames), model.predict(frames), rtol=1e-5, atol=1e-6)


def test_image_value_function_reads_state_frames(model):
    class S:
        def __init__(self, frame):
            self.frame = frame

    states = [S(np.zeros((16, 16, 3), np.uint8)), S(np.full((16, 16, 3), 255, np.uint8))]
    v = ImageValueFunction(model)(states)
    assert v.shape == (2,) and v.dtype == np.float64
