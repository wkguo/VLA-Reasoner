"""The core (KDE, MCTS, reasoner, interfaces, data, policies' base) must import without heavy dependencies."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEAVY = ["torch", "torchvision", "tensorflow", "jax", "transformers", "h5py", "simpler_env", "octo", "libero",
         "openpi_client", "ivideogpt"]

CODE = f"""
import sys
import vla_reasoner, vla_reasoner.kde, vla_reasoner.mcts, vla_reasoner.reasoner, vla_reasoner.sampling
import vla_reasoner.value, vla_reasoner.world_model, vla_reasoner.world_model.oracle, vla_reasoner.data
import vla_reasoner.policies, vla_reasoner.deploy
loaded = [m for m in {HEAVY!r} if m in sys.modules]
assert not loaded, loaded
print("ok")
"""


def test_core_imports_without_heavy_dependencies():
    out = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
