"""Pack per-particle state into a GPU vertex buffer."""

from __future__ import annotations

import numpy as np

from galaxy import Galaxy


class ParticleSystem:
    """CPU-side particle store. Motion is evaluated on the GPU from rest parameters."""

    def __init__(self, galaxy: Galaxy) -> None:
        self.galaxy = galaxy
        self.data = galaxy.generate()
        self.count = int(self.data["theta"].shape[0])
        self.buffer = self.pack()

    def pack(self) -> np.ndarray:
        """Interleaved 16 floats per particle (4 vec4 attributes)."""
        d = self.data
        n = self.count
        buf = np.empty((n, 16), dtype=np.float32)
        buf[:, 0] = d["theta"]
        buf[:, 1] = d["radius"]
        buf[:, 2] = d["z"]
        buf[:, 3] = d["ptype"]
        buf[:, 4:7] = d["rgb"]
        buf[:, 7] = d["size"]
        buf[:, 8] = d["brightness"]
        buf[:, 9] = d["alpha"]
        buf[:, 10] = d["delay"]
        buf[:, 11] = d["omega"]
        buf[:, 12] = d["phase"]
        buf[:, 13] = d["twinkle"]
        buf[:, 14] = d["seed"]
        buf[:, 15] = d["spread"]
        return np.ascontiguousarray(buf)

    def rebuild(self) -> None:
        self.data = self.galaxy.generate()
        self.count = int(self.data["theta"].shape[0])
        self.buffer = self.pack()
