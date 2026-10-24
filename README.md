# Heart of the Universe

A live, GPU-accelerated particle simulation of a spiral galaxy whose outer silhouette is a heart — recreating the attached cosmic artwork as thousands of independently moving points of light.

This is **not** a video, an image slideshow, or a textured sprite. The entire image is generated every frame from particle geometry and GLSL shaders.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The window opens immediately and the galaxy begins forming from a black field of stars.

Target composition is **9:16** (1080×1920). The window is scaled to fit your display and letterboxed if you resize it.

## Controls

| Key | Action |
| --- | --- |
| `SPACE` | Pause / resume |
| `R` | Restart the formation sequence |
| `UP` / `DOWN` | Animation speed |
| `C` | Colour intensity |
| `Z` | Cycle zoom |
| `H` | Hide / show HUD |
| `S` | Save `screenshot.png` |
| `ESC` | Exit |

## How the heart galaxy is built

1. A parametric heart curve defines the outer boundary:
   `x = 16 sin³(t)`, `y = 13 cos(t) − 5 cos(2t) − 2 cos(3t) − cos(4t)`.
2. Two primary logarithmic spiral arms plus three fainter filaments are generated in that parameter space, so the arms themselves follow the lobes, cleft and pointed tip — they are not a round galaxy clipped by a heart.
3. Particle classes: distant field stars, a dense gold-white core, spiral-arm stars, soft nebula clouds, a luminous rim, and diffraction-spike sparkles.
4. Motion (orbit, formation outflow, twinkle, core pulse, turbulence) is evaluated on the GPU from each particle’s rest parameters.

## Stack

- Python, NumPy
- Pygame window + input
- ModernGL + GLSL 330 (point sprites, additive blending, bloom)

## Project layout

```
main.py          entry point, controls, live loop
galaxy.py        heart equation, spiral field, colour distribution
particles.py     packed GPU vertex buffer
renderer.py      shaders, FBOs, bloom composite, HUD
camera.py        9:16 orthographic camera, zoom, cinematic sway
settings.py      counts, timing, look
shaders/         particle, glow, composite
```
