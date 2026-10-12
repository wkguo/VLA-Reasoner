"""End-to-end sanity check on the toy corridor task (NOT a paper result)."""

import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def toy():
    spec = importlib.util.spec_from_file_location("toy_reach", os.path.join(ROOT, "examples", "toy_reach.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reasoner_improves_biased_policy(toy):
    out = toy.run(episodes=30, alpha=0.6)
    assert out["reasoner_success"] >= out["base_success"] + 0.3


def test_alpha_one_equals_base(toy):
    out = toy.run(episodes=10, alpha=1.0)
    assert out["reasoner_success"] == out["base_success"]
