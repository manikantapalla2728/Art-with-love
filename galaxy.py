"""Heart-shaped spiral galaxy: polar log-spirals warped onto a heart."""

from __future__ import annotations

import numpy as np

import settings as S

TWO_PI = 2.0 * np.pi


def heart_curve(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parametric heart used as the galaxy's outer silhouette."""
    t = np.asarray(t, dtype=np.float64)
    sin_t = np.sin(t)
    x = 16.0 * sin_t**3
    y = (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2.0 * t)
        - 2.0 * np.cos(3.0 * t)
        - np.cos(4.0 * t)
    )
    x = x * (S.HEART_SCALE_X / 16.5)
    y = (y + 6.0) * (S.HEART_SCALE_Y / 16.0) + S.HEART_Y_OFFSET
    return x.astype(np.float32), y.astype(np.float32)


def build_heart_lut(n: int = S.HEART_LUT_SIZE) -> np.ndarray:
    """Radius of the heart boundary vs angle in [-pi, pi), for GPU sampling."""
    t = np.linspace(0.0, TWO_PI, n * 2, endpoint=False)
    hx, hy = heart_curve(t)
    ang = np.arctan2(hy.astype(np.float64), hx.astype(np.float64))
    rad = np.hypot(hx.astype(np.float64), hy.astype(np.float64))
    order = np.argsort(ang)
    ang_s = ang[order]
    rad_s = rad[order]
    ang_ext = np.concatenate([ang_s - TWO_PI, ang_s, ang_s + TWO_PI])
    rad_ext = np.concatenate([rad_s, rad_s, rad_s])
    grid = np.linspace(-np.pi, np.pi, n, endpoint=False)
    lut = np.interp(grid, ang_ext, rad_ext).astype(np.float32)
    return lut


_HEART_LUT = build_heart_lut()


def heart_radius(theta: np.ndarray) -> np.ndarray:
    """Interpolate heart boundary radius at polar angles."""
    n = _HEART_LUT.shape[0]
    u = (np.asarray(theta, dtype=np.float64) + np.pi) / TWO_PI
    u = np.mod(u, 1.0)
    idx = u * n
    i0 = np.floor(idx).astype(np.int64) % n
    i1 = (i0 + 1) % n
    f = (idx - np.floor(idx)).astype(np.float32)
    return _HEART_LUT[i0] * (1.0 - f) + _HEART_LUT[i1] * f


def warp_to_heart(radius: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-disk polar coords → heart disk. Inner core stays nearly circular."""
    r = np.clip(np.asarray(radius, dtype=np.float32), 0.0, 1.0)
    th = np.asarray(theta, dtype=np.float32)
    Rh = heart_radius(th)
    # Inner disk stays a round vortex; only the outer envelope becomes the heart.
    blend = np.clip((r - 0.16) / 0.50, 0.0, 1.0)
    blend = np.power(blend, 0.82)
    circular = 0.70
    R = (1.0 - blend) * circular + blend * Rh
    x = r * R * np.cos(th)
    y = r * R * np.sin(th)
    return x.astype(np.float32), y.astype(np.float32)


def _palette(theta: np.ndarray, u: np.ndarray, arm: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Arm-carried colour: cool and warm filaments weave through the heart."""
    gold = np.array([1.00, 0.78, 0.22], np.float32)
    white = np.array([1.00, 0.96, 0.88], np.float32)
    yellow = np.array([1.00, 0.62, 0.12], np.float32)
    cyan = np.array([0.08, 0.78, 1.00], np.float32)
    teal = np.array([0.05, 0.92, 0.76], np.float32)
    blue = np.array([0.18, 0.38, 1.00], np.float32)
    violet = np.array([0.58, 0.18, 1.00], np.float32)
    magenta = np.array([1.00, 0.12, 0.62], np.float32)
    pink = np.array([1.00, 0.34, 0.72], np.float32)
    crimson = np.array([0.92, 0.10, 0.32], np.float32)

    cth = np.cos(theta)
    sth = np.sin(theta)
    right = np.clip(0.5 + 0.55 * cth, 0.0, 1.0)
    left = 1.0 - right
    top = np.clip(0.5 + 0.55 * sth, 0.0, 1.0)
    bot = 1.0 - top
    cool_arm = ((arm % 2) == 0).astype(np.float32)

    # Colour rides the spiral arm, with a lighter spatial tint so both hues appear on each side.
    streak = 0.5 + 0.5 * np.sin(theta * 2.8 + np.log(u + 0.04) * 6.0 + noise * 8.0)
    cool = (
        cyan[None, :] * (0.55 + 0.25 * streak)[:, None]
        + teal[None, :] * (0.28 + 0.12 * left)[:, None]
        + blue[None, :] * (0.16 * top)[:, None]
        + violet[None, :] * (0.14 * top)[:, None]
    )
    warm = (
        magenta[None, :] * (0.50 + 0.22 * streak)[:, None]
        + pink[None, :] * (0.28 + 0.10 * right)[:, None]
        + crimson[None, :] * (0.12 * bot)[:, None]
        + gold[None, :] * (0.22 * (0.4 + 0.6 * right))[:, None]
        + yellow[None, :] * (0.10 * streak)[:, None]
    )
    arm_mix = np.clip(0.62 * cool_arm + 0.38 * left, 0.0, 1.0)
    mid = cool * arm_mix[:, None] + warm * (1.0 - arm_mix)[:, None]
    mid *= (0.84 + 0.32 * noise)[:, None]

    inner = white[None, :] * 0.22 + gold[None, :] * 0.90 + yellow[None, :] * 0.42
    inner = inner * (0.90 + 0.16 * noise)[:, None]

    w_core = np.clip(1.0 - u / 0.20, 0.0, 1.0) ** 1.5
    w_gold = np.clip(1.0 - np.abs(u - 0.26) / 0.24, 0.0, 1.0) ** 1.05
    w_mid = np.clip((u - 0.12) / 0.50, 0.0, 1.0) * np.clip((1.05 - u) / 0.40, 0.0, 1.0)
    w_rim = np.clip((u - 0.60) / 0.40, 0.0, 1.0)

    rim = (
        cyan[None, :] * (0.55 * left + 0.20 * cool_arm)[:, None]
        + magenta[None, :] * (0.55 * right + 0.20 * (1.0 - cool_arm))[:, None]
        + gold[None, :] * (0.45 * right * (0.35 + 0.65 * bot))[:, None]
        + violet[None, :] * (0.22 * top)[:, None]
        + teal[None, :] * (0.18 * left)[:, None]
    )

    rgb = (
        inner * (1.20 * w_core)[:, None]
        + gold[None, :] * (0.95 * w_gold)[:, None]
        + yellow[None, :] * (0.32 * w_gold)[:, None]
        + mid * (0.80 + 0.55 * w_mid)[:, None]
        + rim * (1.00 * w_rim)[:, None]
    )
    return np.clip(rgb, 0.0, 1.7).astype(np.float32)


class Galaxy:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng(0xC0FFEE)
        self.formation_time = 0.0
        self.sim_time = 0.0
        self.speed = S.DEFAULT_SPEED
        self.color_intensity = S.COLOR_INTENSITY_DEFAULT
        self.paused = False

    def restart(self) -> None:
        self.formation_time = 0.0
        self.sim_time = 0.0

    def step(self, dt: float) -> None:
        if self.paused:
            return
        dt *= self.speed
        self.sim_time += dt
        self.formation_time += dt

    def generate(self) -> dict[str, np.ndarray]:
        parts = [
            self._background(S.BACKGROUND_COUNT),
            self._core(S.CORE_COUNT),
            self._inner(S.INNER_COUNT),
            self._spiral(S.SPIRAL_COUNT),
            self._nebula(S.NEBULA_COUNT),
            self._lobes(S.LOBE_COUNT),
            self._haze(S.HAZE_COUNT),
            self._rim(S.RIM_COUNT),
            self._sparkles(S.SPARKLE_COUNT),
        ]
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}

    def _alloc(self, n: int, ptype: float) -> dict[str, np.ndarray]:
        return {
            "theta": np.zeros(n, np.float32),
            "radius": np.zeros(n, np.float32),
            "z": np.zeros(n, np.float32),
            "ptype": np.full(n, ptype, np.float32),
            "rgb": np.zeros((n, 3), np.float32),
            "size": np.zeros(n, np.float32),
            "brightness": np.zeros(n, np.float32),
            "alpha": np.zeros(n, np.float32),
            "delay": np.zeros(n, np.float32),
            "omega": np.zeros(n, np.float32),
            "phase": np.zeros(n, np.float32),
            "twinkle": np.zeros(n, np.float32),
            "seed": np.zeros(n, np.float32),
            "spread": np.zeros(n, np.float32),
        }

    def _common(self, d: dict[str, np.ndarray], n: int) -> None:
        rng = self.rng
        d["seed"] = rng.random(n, dtype=np.float32)
        d["phase"] = rng.uniform(0.0, TWO_PI, n).astype(np.float32)
        d["twinkle"] = (0.5 + 2.2 * rng.random(n)).astype(np.float32)
        d["z"] = rng.normal(0.0, S.DISK_THICKNESS, n).astype(np.float32)

    def _arm_polar(self, n: int, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = self.rng
        # Skew outward so the heart lobes and tip fill instead of a round mid-disk.
        u = rng.beta(1.25, 1.05, n).astype(np.float32)
        u = np.clip(u * 0.96 + 0.03, 0.02, 0.995)
        # Two bright primary arms plus three fainter filaments.
        primary = rng.random(n) < 0.70
        arm = np.zeros(n, dtype=np.int64)
        n_pri = int(primary.sum())
        n_sec = n - n_pri
        if n_pri:
            arm[primary] = rng.integers(0, S.NUM_ARMS, n_pri)
        if n_sec:
            arm[~primary] = rng.integers(0, 5, n_sec)
        n_arms = np.where(primary, float(S.NUM_ARMS), 5.0)
        spread = np.empty(n, dtype=np.float64)
        if n_pri:
            spread[primary] = rng.vonmises(0.0, float(kappa), n_pri)
        if n_sec:
            spread[~primary] = rng.vonmises(0.0, max(float(kappa) * 0.38, 1.4), n_sec)
        pitch = np.where(u < 0.34, S.SPIRAL_PITCH * 0.78, S.SPIRAL_PITCH)
        theta = (
            arm.astype(np.float64) * (TWO_PI / n_arms)
            + np.log(u + 0.022) / pitch
            + S.SPIRAL_PHASE
            + spread
        )
        return theta.astype(np.float32), u, arm

    def _spiral(self, n: int) -> dict[str, np.ndarray]:
        rng = self.rng
        d = self._alloc(n, 2.0)
        self._common(d, n)
        theta, u, arm = self._arm_polar(n, S.ARM_KAPPA)
        d["theta"], d["radius"] = theta, u
        d["rgb"] = _palette(theta, u, arm, d["seed"])
        d["size"] = (1.5 + 3.1 * (1.0 - 0.50 * u) + 1.2 * rng.random(n)).astype(np.float32)
        d["brightness"] = (0.42 + 0.48 * rng.random(n) * (1.05 - 0.30 * u)).astype(np.float32)
        d["alpha"] = (0.32 + 0.40 * rng.random(n)).astype(np.float32)
        d["delay"] = (
            S.FORMATION_SPIRAL_START + u * S.FORMATION_SPIRAL_SPAN + 0.6 * rng.random(n)
        ).astype(np.float32)
        d["omega"] = (S.BASE_ROTATION * (0.50 + 0.85 / (u + 0.16))).astype(np.float32)
        d["z"] *= (1.0 - 0.5 * u).astype(np.float32)
        return d

    def _core(self, n: int) -> dict[str, np.ndarray]:
        rng = self.rng
        d = self._alloc(n, 3.0)
        self._common(d, n)
        u = (rng.random(n) ** 2.6) * 0.16
        theta = rng.random(n) * TWO_PI + np.log(u + 0.018) / 0.42
        d["theta"] = theta.astype(np.float32)
        d["radius"] = u.astype(np.float32)
        mix = (rng.random(n) ** 0.55).astype(np.float32)
        white = np.array([1.0, 0.97, 0.90], np.float32)
        gold = np.array([1.0, 0.80, 0.28], np.float32)
        orange = np.array([1.0, 0.48, 0.10], np.float32)
        d["rgb"] = (
            white[None, :] * (1.0 - 0.62 * mix)[:, None]
            + gold[None, :] * (0.85 * mix)[:, None]
            + orange[None, :] * (0.40 * mix * (u / 0.16))[:, None]
        )
        d["size"] = (1.4 + 2.6 * (1.0 - u / 0.16) + 0.8 * rng.random(n)).astype(np.float32)
        d["brightness"] = (0.70 + 0.75 * (1.0 - u / 0.16)).astype(np.float32)
        d["alpha"] = (0.42 + 0.32 * rng.random(n)).astype(np.float32)
        d["delay"] = rng.uniform(S.FORMATION_CORE_START, S.FORMATION_CORE_END, n).astype(np.float32)
        d["omega"] = (S.BASE_ROTATION * (2.8 + 4.2 / (u + 0.04))).astype(np.float32)
        d["z"] = rng.normal(0.0, 0.016, n).astype(np.float32)
        return d

    def _inner(self, n: int) -> dict[str, np.ndarray]:
        """Gold-dust inner spiral so the vortex reads as warm, not a white disk."""
        rng = self.rng
        d = self._alloc(n, 2.0)
        self._common(d, n)
        theta, u, arm = self._arm_polar(n, S.ARM_KAPPA * 1.35)
        u = (u ** 1.35) * 0.42
        d["theta"], d["radius"] = theta, u.astype(np.float32)
        gold = np.array([1.00, 0.76, 0.22], np.float32)
        yellow = np.array([1.00, 0.58, 0.12], np.float32)
        pink = np.array([1.00, 0.32, 0.55], np.float32)
        mix = rng.random(n).astype(np.float32)
        d["rgb"] = (
            gold[None, :] * (0.75 + 0.25 * (1.0 - mix))[:, None]
            + yellow[None, :] * (0.35 * mix)[:, None]
            + pink[None, :] * (0.18 * mix * (arm % 2))[:, None]
        )
        d["rgb"] = np.clip(d["rgb"], 0.0, 1.6)
        d["size"] = (1.6 + 2.8 * rng.random(n)).astype(np.float32)
        d["brightness"] = (0.50 + 0.40 * rng.random(n)).astype(np.float32)
        d["alpha"] = (0.28 + 0.30 * rng.random(n)).astype(np.float32)
        d["delay"] = (S.FORMATION_SPIRAL_START + u * 4.0 + 0.4 * rng.random(n)).astype(np.float32)
        d["omega"] = (S.BASE_ROTATION * (1.1 + 1.6 / (u + 0.08))).astype(np.float32)
        d["z"] = rng.normal(0.0, 0.022, n).astype(np.float32)
        return d

    def _nebula(self, n: int) -> dict[str, np.ndarray]:
        rng = self.rng
        d = self._alloc(n, 1.0)
        self._common(d, n)
        theta, u, arm = self._arm_polar(n, S.ARM_KAPPA * 0.55)
        d["theta"], d["radius"] = theta, u
        d["rgb"] = _palette(theta, u, arm, d["seed"]) * 1.20
        d["size"] = (22.0 + 56.0 * rng.random(n) * (0.30 + 0.70 * u)).astype(np.float32)
        d["brightness"] = (0.26 + 0.32 * rng.random(n)).astype(np.float32)
        d["alpha"] = (0.16 + 0.22 * rng.random(n)).astype(np.float32)
        d["delay"] = (S.FORMATION_NEBULA_START + u * 6.5 + rng.random(n) * 1.8).astype(np.float32)
        d["omega"] = (S.BASE_ROTATION * (0.42 + 0.45 / (u + 0.22))).astype(np.float32)
        d["z"] *= 1.35
        return d

    def _lobes(self, n: int) -> dict[str, np.ndarray]:
        """Dense nebula packed into the two upper lobes and the pointed tip."""
        rng = self.rng
        d = self._alloc(n, 1.0)
        self._common(d, n)
        pick = rng.random(n)
        theta = np.empty(n, np.float32)
        # Right lobe, left lobe, bottom tip, cleft shoulders.
        n_r = pick < 0.32
        n_l = (pick >= 0.32) & (pick < 0.64)
        n_t = (pick >= 0.64) & (pick < 0.88)
        n_c = pick >= 0.88
        theta[n_r] = rng.normal(0.78, 0.28, int(n_r.sum())).astype(np.float32)
        theta[n_l] = rng.normal(2.36, 0.28, int(n_l.sum())).astype(np.float32)
        theta[n_t] = rng.normal(-1.57, 0.22, int(n_t.sum())).astype(np.float32)
        theta[n_c] = rng.normal(1.57, 0.18, int(n_c.sum())).astype(np.float32)
        u = (0.58 + 0.40 * rng.beta(1.1, 0.9, n)).astype(np.float32)
        u[n_t] = (0.70 + 0.28 * rng.random(int(n_t.sum()))).astype(np.float32)
        arm = np.where(n_l, 0, 1).astype(np.int64)
        d["theta"], d["radius"] = theta, np.clip(u, 0.45, 0.995)
        d["rgb"] = _palette(d["theta"], d["radius"], arm, d["seed"]) * 1.18
        d["size"] = (16.0 + 48.0 * rng.random(n)).astype(np.float32)
        d["brightness"] = (0.20 + 0.26 * rng.random(n)).astype(np.float32)
        d["alpha"] = (0.13 + 0.18 * rng.random(n)).astype(np.float32)
        d["delay"] = (S.FORMATION_NEBULA_START + 1.2 + d["radius"] * 5.5).astype(np.float32)
        d["omega"] = (S.BASE_ROTATION * (0.38 + 0.4 / (d["radius"] + 0.25))).astype(np.float32)
        d["z"] *= 1.25
        return d

    def _haze(self, n: int) -> dict[str, np.ndarray]:
        """Soft inter-arm fill so the heart reads as a nebula, not hollow lanes."""
        rng = self.rng
        d = self._alloc(n, 1.0)
        self._common(d, n)
        theta, u, arm = self._arm_polar(n, 3.2)
        d["theta"], d["radius"] = theta, u
        d["rgb"] = _palette(theta, u, arm, d["seed"]) * 0.70
        d["size"] = (24.0 + 42.0 * rng.random(n)).astype(np.float32)
        d["brightness"] = (0.07 + 0.10 * rng.random(n)).astype(np.float32)
        d["alpha"] = (0.05 + 0.08 * rng.random(n)).astype(np.float32)
        d["delay"] = (S.FORMATION_NEBULA_START + 1.5 + u * 6.0).astype(np.float32)
        d["omega"] = (S.BASE_ROTATION * (0.4 + 0.4 / (u + 0.25))).astype(np.float32)
        return d

    def _rim(self, n: int) -> dict[str, np.ndarray]:
        rng = self.rng
        d = self._alloc(n, 2.0)
        self._common(d, n)
        # Trace the actual heart boundary with extra weight on lobes and tip.
        t = rng.random(n) * TWO_PI
        hx, hy = heart_curve(t)
        theta = np.arctan2(hy, hx)
        u = (0.86 + 0.13 * rng.random(n)).astype(np.float32)
        arm = np.zeros(n, np.int64)
        d["theta"] = theta.astype(np.float32)
        d["radius"] = u
        d["rgb"] = np.clip(_palette(d["theta"], u, arm, d["seed"]) * 1.35, 0.0, 1.7)
        # Golden lower-right ridge from the reference.
        gold_ridge = (theta < -0.15) & (theta > -1.05) & (rng.random(n) < 0.55)
        gold = np.array([1.00, 0.78, 0.28], np.float32)
        d["rgb"][gold_ridge] = gold * (0.85 + 0.35 * d["seed"][gold_ridge])[:, None]
        d["size"] = (2.0 + 5.2 * rng.random(n)).astype(np.float32)
        d["brightness"] = (0.55 + 0.50 * rng.random(n)).astype(np.float32)
        d["alpha"] = (0.30 + 0.38 * rng.random(n)).astype(np.float32)
        d["delay"] = (6.2 + 4.5 * rng.random(n)).astype(np.float32)
        d["omega"] = np.full(n, S.BASE_ROTATION * 0.55, np.float32)
        return d

    def _sparkles(self, n: int) -> dict[str, np.ndarray]:
        rng = self.rng
        d = self._alloc(n, 4.0)
        self._common(d, n)
        theta, u, _arm = self._arm_polar(n, 6.0)
        # A few bright field sparkles outside the heart.
        extra = rng.random(n) > 0.78
        theta = theta.copy()
        u = u.copy()
        theta[extra] = rng.uniform(0.0, TWO_PI, extra.sum()).astype(np.float32)
        u[extra] = rng.uniform(0.2, 1.35, extra.sum()).astype(np.float32)
        d["theta"], d["radius"] = theta, np.clip(u, 0.04, 1.4)
        white = np.array([0.95, 0.97, 1.0], np.float32)
        cyan = np.array([0.65, 0.9, 1.0], np.float32)
        gold = np.array([1.0, 0.9, 0.65], np.float32)
        pick = rng.random(n)
        d["rgb"][:] = white
        d["rgb"][pick > 0.5] = cyan
        d["rgb"][pick > 0.8] = gold
        # A handful of hero starbursts around the silhouette, like the reference.
        hero = rng.random(n) > 0.93
        d["size"] = (4.2 + 9.0 * rng.random(n) ** 1.45).astype(np.float32)
        d["size"][hero] = (14.0 + 18.0 * rng.random(int(hero.sum()))).astype(np.float32)
        d["brightness"] = (0.85 + 1.05 * rng.random(n)).astype(np.float32)
        d["brightness"][hero] = (1.4 + 0.8 * rng.random(int(hero.sum()))).astype(np.float32)
        d["alpha"] = (0.50 + 0.40 * rng.random(n)).astype(np.float32)
        d["delay"] = (5.2 + 6.5 * rng.random(n)).astype(np.float32)
        d["omega"] = (S.BASE_ROTATION * (0.45 + 0.7 / (np.clip(u, 0.05, 1.0) + 0.2))).astype(
            np.float32
        )
        return d

    def _background(self, n: int) -> dict[str, np.ndarray]:
        rng = self.rng
        d = self._alloc(n, 0.0)
        self._common(d, n)
        n_field = int(n * 0.87)
        x = rng.uniform(-S.VIEW_WIDTH * 0.58, S.VIEW_WIDTH * 0.58, n)
        y = rng.uniform(-S.VIEW_HEIGHT * 0.48, S.VIEW_HEIGHT * 0.48, n)
        d["theta"] = x.astype(np.float32)
        d["radius"] = y.astype(np.float32)
        d["z"] = rng.uniform(-0.9, 0.9, n).astype(np.float32)
        n_gals = n - n_field
        if n_gals > 0:
            centres = np.array(
                [
                    [-0.82, 0.78],
                    [0.86, 0.58],
                    [0.74, -0.92],
                    [-0.76, -0.74],
                    [0.58, -1.12],
                ],
                dtype=np.float32,
            )
            idx = rng.integers(0, len(centres), n_gals)
            ang = rng.random(n_gals) * TWO_PI
            rad = rng.random(n_gals) ** 1.8 * 0.09
            d["theta"][n_field:] = centres[idx, 0] + rad * np.cos(ang) * 2.3
            d["radius"][n_field:] = centres[idx, 1] + rad * np.sin(ang) * 0.5
        d["rgb"][:] = np.array([0.72, 0.82, 1.0], np.float32)
        tint = rng.random(n)
        d["rgb"][tint > 0.72] = np.array([1.0, 0.90, 0.72], np.float32)
        d["rgb"][tint > 0.9] = np.array([0.82, 0.68, 1.0], np.float32)
        d["size"] = (1.15 + 2.3 * rng.random(n) ** 2).astype(np.float32)
        d["brightness"] = (0.18 + 0.70 * rng.random(n) ** 1.6).astype(np.float32)
        d["alpha"] = (0.35 + 0.5 * rng.random(n)).astype(np.float32)
        d["delay"] = (rng.random(n) * S.FORMATION_STAR_END).astype(np.float32)
        d["omega"] = rng.uniform(-0.008, 0.008, n).astype(np.float32)
        return d
