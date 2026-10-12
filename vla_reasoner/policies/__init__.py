"""Frozen VLA adapters. Each adapter imports its model stack (jax/octo, transformers, openpi_client, ...) lazily.

  base.VLAPolicy                         -- interface used by the LIBERO and real-world loops
  octo_simpler.make_octo_policy          -- Octo via SimplerEnv's OctoInference (nakamotoo fork used by V-GPS)
  spatialvla_simpler.make_spatialvla_policy -- SpatialVLA via DelinQu/SimplerEnv-OpenVLA
  openvla.OpenVLAPolicy                  -- OpenVLA (LIBERO OpenVLA-SFT, real world)
  openpi_client.OpenPIClientPolicy       -- pi0-FAST served by openpi (real world)
"""

from vla_reasoner.policies.base import VLAPolicy

__all__ = ["VLAPolicy"]
