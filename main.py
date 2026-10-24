"""Heart of the Universe — live GPU particle galaxy.

Run:
    python main.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pygame
import moderngl

import settings as S
from camera import Camera
from galaxy import Galaxy
from particles import ParticleSystem
from renderer import Renderer


def _fit_window() -> tuple[int, int]:
    info = pygame.display.Info()
    max_w = max(info.current_w - S.WINDOW_MARGIN, 360)
    max_h = max(info.current_h - S.WINDOW_MARGIN, 640)
    w, h = S.TARGET_WIDTH, S.TARGET_HEIGHT
    scale = min(max_w / w, max_h / h, 1.0)
    return max(int(w * scale), 270), max(int(h * scale), 480)


def _make_gl_window(
    size: tuple[int, int], resizable: bool = True, vsync: int = 1
) -> pygame.Surface:
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if resizable:
        flags |= pygame.RESIZABLE
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 16)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    try:
        return pygame.display.set_mode(size, flags, vsync=vsync)
    except pygame.error:
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
        return pygame.display.set_mode(size, flags)


def _record_mp4(renderer: Renderer, galaxy: Galaxy, camera: Camera, size: tuple[int, int], path: Path, seconds: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    fps = S.TARGET_FPS
    n_frames = max(int(round(seconds * fps)), 1)
    dt = 1.0 / fps
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{S.TARGET_WIDTH}x{S.TARGET_HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "vflip",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.2",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=1024 * 1024)
    assert proc.stdin is not None
    galaxy.restart()
    try:
        for i in range(n_frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
            galaxy.step(dt)
            camera.update(galaxy.sim_time, galaxy.paused)
            renderer.render(galaxy, float(fps), size)
            proc.stdin.write(renderer.read_frame_rgb())
            if i % 10 == 0:
                pygame.display.flip()
            if i == 0 or (i + 1) % fps == 0 or i + 1 == n_frames:
                print(f"recording {i + 1}/{n_frames}  {path.name}", flush=True)
        proc.stdin.close()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with {code}")
    except BaseException:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()
        raise


def run(
    selftest: bool = False,
    screenshot: Path | None = None,
    seconds: float = 0.0,
    record: Path | None = None,
) -> int:
    pygame.init()
    pygame.font.init()
    size = _fit_window()
    _make_gl_window(size, vsync=0 if record else 1)
    pygame.display.set_caption(S.WINDOW_TITLE)

    ctx = moderngl.create_context()
    ctx.gc_mode = "auto"

    galaxy = Galaxy(np.random.default_rng(0x5EED))
    particles = ParticleSystem(galaxy)
    camera = Camera()
    renderer = Renderer(ctx, particles, camera)
    renderer.resize(*size)

    clock = pygame.time.Clock()
    running = True
    color_steps = (0.7, 1.0, 1.35, 1.7)
    color_i = 1

    if selftest:
        from galaxy import warp_to_heart

        galaxy.formation_time = S.FORMATION_TOTAL
        galaxy.sim_time = 0.6
        d = particles.data
        mask = d["ptype"] > 0.5
        x, y = warp_to_heart(d["radius"][mask], d["theta"][mask])
        print(
            "pos",
            float(x.min()),
            float(x.max()),
            float(y.min()),
            float(y.max()),
            "n",
            particles.count,
        )
        print("gl", ctx.info.get("GL_RENDERER"), ctx.info.get("GL_VERSION"))
        renderer.render(galaxy, 60.0, size)
        out = screenshot or (S.ROOT / "screenshot.png")
        renderer.capture_png(out)
        renderer.save_scene(S.ROOT / "scene_debug.png")
        pygame.display.flip()
        print(f"selftest ok  particles={particles.count}  screenshot={out}")
        pygame.quit()
        return 0

    if record is not None:
        duration = seconds if seconds > 0.0 else S.VIDEO_SECONDS
        try:
            _record_mp4(renderer, galaxy, camera, size, record, duration)
            print(f"wrote {record}  {S.TARGET_WIDTH}x{S.TARGET_HEIGHT}@{S.TARGET_FPS}  {duration:.1f}s")
        except KeyboardInterrupt:
            print("recording cancelled")
            pygame.quit()
            return 1
        pygame.quit()
        return 0

    elapsed = 0.0
    while running:
        dt = clock.tick(S.TARGET_FPS) / 1000.0
        dt = min(dt, 1.0 / 20.0)
        fps = clock.get_fps()
        elapsed += dt
        if seconds > 0.0 and elapsed >= seconds:
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                size = (max(event.w, 270), max(event.h, 480))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    galaxy.paused = not galaxy.paused
                elif event.key == pygame.K_r:
                    galaxy.restart()
                elif event.key == pygame.K_UP:
                    galaxy.speed = min(S.MAX_SPEED, galaxy.speed + 0.15)
                elif event.key == pygame.K_DOWN:
                    galaxy.speed = max(S.MIN_SPEED, galaxy.speed - 0.15)
                elif event.key == pygame.K_c:
                    color_i = (color_i + 1) % len(color_steps)
                    galaxy.color_intensity = color_steps[color_i]
                elif event.key == pygame.K_z:
                    camera.cycle_zoom()
                elif event.key == pygame.K_h:
                    renderer.show_hud = not renderer.show_hud
                elif event.key == pygame.K_s:
                    renderer.capture_png(S.ROOT / "screenshot.png")
                elif event.key == pygame.K_v:
                    out = S.ROOT / "heart_of_the_universe.mp4"
                    pygame.display.set_caption("Recording 1080x1920…")
                    _record_mp4(renderer, galaxy, camera, size, out, S.VIDEO_SECONDS)
                    pygame.display.set_caption(S.WINDOW_TITLE)
                    print(f"wrote {out}")

        galaxy.step(dt)
        camera.update(galaxy.sim_time, galaxy.paused)
        surf = pygame.display.get_surface()
        if surf is not None:
            size = surf.get_size()
        renderer.render(galaxy, fps, size)
        pygame.display.flip()

    pygame.quit()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart of the Universe")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Render one formed-galaxy frame, save screenshot.png, exit",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Live-loop duration, or video length with --record (0 = forever / default video length)",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=str(S.ROOT / "heart_of_the_universe.mp4"),
        default=None,
        help="Render a 1080x1920 quality MP4 and exit",
    )
    args = parser.parse_args()
    out = Path(args.record) if args.record else None
    sys.exit(run(selftest=args.selftest, seconds=args.seconds, record=out))


if __name__ == "__main__":
    main()
