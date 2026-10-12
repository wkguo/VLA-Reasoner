"""pi0-FAST (or any openpi policy) queried over openpi's websocket protocol (real-world experiments, Table II).

Serve the fine-tuned checkpoint with openpi's `scripts/serve_policy.py`; this client only needs the lightweight
`openpi-client` package (Physical-Intelligence/openpi, packages/openpi-client; pinned to commit 215abfb in
requirements/real_openpi.txt).
The observation dict must use the keys of the served policy config (e.g. "observation/image",
"observation/wrist_image", "observation/state"); the robot driver is responsible for building it.
The paper executes 5 actions of each pi0-FAST chunk (Sec. IV-B).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from vla_reasoner.policies.base import VLAPolicy


class OpenPIClientPolicy(VLAPolicy):
    def __init__(self, host: str = "localhost", port: Optional[int] = 8000, exec_chunk: int = 5,
                 action_dim: Optional[int] = None, api_key: Optional[str] = None):
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=api_key)
        self.exec_chunk = exec_chunk
        self.action_dim = action_dim  # set from the first response if None
        self.instruction = ""

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.client.reset()

    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        obs = dict(observation)
        obs.setdefault("prompt", self.instruction)
        actions = np.asarray(self.client.infer(obs)["actions"], dtype=np.float64)
        if len(actions) < self.exec_chunk:
            raise ValueError(f"Policy returned {len(actions)} actions but exec_chunk={self.exec_chunk}.")
        if self.action_dim is None:
            self.action_dim = actions.shape[-1]
        return actions[: self.exec_chunk, : self.action_dim]
