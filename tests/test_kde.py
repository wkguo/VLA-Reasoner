"""Tests for the KDE action prior."""

import math

import numpy as np
import pytest

from vla_reasoner.kde import ActionKDE, bandwidth_factor, nearest_k
from vla_reasoner.sampling import GaussianNoiseSampler, KDETopKSampler


def _grid_integral_2d(kde, lo, hi, n=601):
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    p = kde.density(np.stack([xx.ravel(), yy.ravel()], 1)).reshape(n, n)
    return np.trapezoid(np.trapezoid(p, ys, axis=1), xs) if hasattr(np, "trapezoid") else np.trapz(np.trapz(p, ys, axis=1), xs)


@pytest.mark.parametrize("standardize", [True, False])
def test_density_integrates_to_one_1d(standardize):
    rng = np.random.default_rng(0)
    data = rng.normal(2.0, 0.5, size=(300, 1))
    kde = ActionKDE(bandwidth="silverman", standardize=standardize).fit(data)
    xs = np.linspace(-2, 6, 4001)
    p = kde.density(xs[:, None])
    integral = np.sum(p) * (xs[1] - xs[0])
    assert abs(integral - 1.0) < 1e-3


@pytest.mark.parametrize("standardize", [True, False])
def test_density_integrates_to_one_2d(standardize):
    rng = np.random.default_rng(1)
    data = np.stack([rng.normal(0, 0.05, 400), rng.normal(1, 0.8, 400)], 1)  # very different scales
    kde = ActionKDE(bandwidth="scott", standardize=standardize).fit(data)
    integral = _grid_integral_2d(kde, (-1.5, -4.0), (1.5, 6.0))
    assert abs(integral - 1.0) < 1e-2


def test_log_density_matches_direct_formula():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(50, 3)) * np.array([1.0, 0.1, 3.0])
    kde = ActionKDE(bandwidth=0.4, standardize=True).fit(data)
    q = rng.normal(size=(4, 3))
    z_data = (data - data.mean(0)) / data.std(0)
    z_q = (q - data.mean(0)) / data.std(0)
    h = 0.4
    k = np.exp(-0.5 * ((z_q[:, None] - z_data[None]) ** 2).sum(-1) / h**2) / (2 * math.pi * h**2) ** 1.5
    expected = np.log(k.mean(1)) - np.log(data.std(0)).sum()
    np.testing.assert_allclose(kde.log_density(q), expected, rtol=1e-5, atol=1e-6)


def test_log_density_is_stable_far_from_data():
    kde = ActionKDE(bandwidth=0.1).fit(np.random.default_rng(3).normal(size=(100, 2)))
    lp = kde.log_density(np.array([[1e3, -1e3]]))
    assert np.isfinite(lp).all() and lp[0] < -1e5


def test_log_density_blocking_is_exact():
    rng = np.random.default_rng(4)
    kde = ActionKDE().fit(rng.normal(size=(1000, 4)))
    q = rng.normal(size=(7, 4))
    np.testing.assert_allclose(kde.log_density(q, block=37), kde.log_density(q), rtol=1e-6)


def test_bandwidth_rules():
    assert bandwidth_factor("scott", 1000, 2) == pytest.approx(1000 ** (-1 / 6))
    assert bandwidth_factor("silverman", 1000, 2) == pytest.approx((1000 * 4 / 4) ** (-1 / 6))
    assert bandwidth_factor(0.3, 10, 2) == 0.3
    assert bandwidth_factor("0.3", 10, 2) == 0.3
    with pytest.raises(ValueError):
        bandwidth_factor("nope", 10, 2)


def test_sample_shape_and_chunk_flattening():
    rng = np.random.default_rng(5)
    chunks = rng.normal(size=(200, 4, 7))  # (n, H, A): chunks treated as single samples
    kde = ActionKDE().fit(chunks)
    assert kde.dim == 28 and kde.action_shape == (4, 7)
    assert kde.sample(13, rng).shape == (13, 28)
    np.testing.assert_allclose(kde.log_density(chunks[:3]), kde.log_density(chunks[:3].reshape(3, 28)))


def test_nearest_k_returns_k_closest_sorted():
    cands = np.array([[0.0, 0.0], [1.0, 0.0], [0.1, 0.1], [5.0, 5.0], [-0.3, 0.0]])
    idx = nearest_k(cands, np.zeros(2), 3)
    assert list(idx) == [0, 2, 4]


def test_top_k_returns_the_k_nearest_of_the_sampled_set():
    data = np.random.default_rng(6).normal(size=(500, 3))
    kde = ActionKDE().fit(data)
    ref = np.array([0.5, -0.2, 0.1])
    chosen, logp = kde.top_k(ref, k=5, num_samples=200, rng=np.random.default_rng(7))
    samples = kde.sample(200, np.random.default_rng(7))  # identical draw
    d_all = np.linalg.norm(kde.to_z(samples) - kde.to_z(ref), axis=1)
    d_chosen = np.linalg.norm(kde.to_z(chosen) - kde.to_z(ref), axis=1)
    np.testing.assert_allclose(np.sort(d_chosen), np.sort(d_all)[:5])
    np.testing.assert_allclose(logp, kde.log_density(chosen))


def test_constant_dimension_does_not_dominate_top_k(tmp_path):
    rng = np.random.default_rng(11)
    data = np.concatenate([rng.normal(size=(500, 2)), np.full((500, 1), 0.5)], axis=1)  # last column constant
    kde = ActionKDE().fit(data)
    assert kde.varying.tolist() == [True, True, False] and kde.std[2] == kde.min_std
    ref = np.array([1.0, -1.0, 0.5 + 1e-3])  # slightly off in the constant column
    samples = kde.sample(1000, np.random.default_rng(12))
    chosen, logp = kde.top_k(ref, k=5, num_samples=1000, rng=np.random.default_rng(12))
    # the chosen candidates are the samples nearest to ref in the varying dimensions ...
    d_other = np.linalg.norm(samples[:, :2] - ref[:2], axis=1)
    assert np.max(np.linalg.norm(chosen[:, :2] - ref[:2], axis=1)) <= np.quantile(d_other, 0.05)
    # ... and keep the data's constant value in the constant dimension
    np.testing.assert_allclose(chosen[:, 2], 0.5, atol=1e-4)
    assert np.isfinite(logp).all()
    # the mask survives save/load
    kde.save(str(tmp_path / "kde.npz"))
    assert ActionKDE.load(str(tmp_path / "kde.npz")).varying.tolist() == [True, True, False]


def test_kde_sampler_checks_dimension():
    kde = ActionKDE().fit(np.random.default_rng(8).normal(size=(50, 4, 7)))
    sampler = KDETopKSampler(kde, k=3, num_samples=20)
    actions, logp = sampler(np.zeros((4, 7)), np.random.default_rng(0))
    assert actions.shape == (3, 28) and logp.shape == (3,)
    with pytest.raises(ValueError, match="KDE dimension"):
        sampler(np.zeros(7), np.random.default_rng(0))


def test_gaussian_noise_sampler():
    s = GaussianNoiseSampler(k=4, std=np.array([0.1, 0.2]))
    a, lp = s(np.array([1.0, 2.0]), np.random.default_rng(0))
    assert a.shape == (4, 2) and lp.shape == (4,)
    eps = (a - np.array([1.0, 2.0])) / np.array([0.1, 0.2])
    expected = -0.5 * (eps**2).sum(1) - np.log(0.1 * 0.2) - math.log(2 * math.pi)
    np.testing.assert_allclose(lp, expected)


def test_save_load_roundtrip(tmp_path):
    kde = ActionKDE(bandwidth="silverman").fit(np.random.default_rng(9).normal(size=(80, 2, 3)))
    path = str(tmp_path / "kde.npz")
    kde.save(path)
    kde2 = ActionKDE.load(path)
    q = np.random.default_rng(10).normal(size=(5, 6))
    np.testing.assert_allclose(kde.log_density(q), kde2.log_density(q), rtol=1e-6)
    assert kde2.action_shape == (2, 3) and kde2.h == pytest.approx(kde.h)
