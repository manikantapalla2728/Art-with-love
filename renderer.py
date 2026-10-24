"""ModernGL particle renderer with additive blending and bloom."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame
import moderngl

import settings as S
from camera import Camera
from galaxy import build_heart_lut
from particles import ParticleSystem


def _load(name: str) -> str:
    return (S.SHADER_DIR / name).read_text(encoding="utf-8")


class Renderer:
    def __init__(self, ctx: moderngl.Context, particles: ParticleSystem, camera: Camera) -> None:
        self.ctx = ctx
        self.particles = particles
        self.camera = camera
        self.show_hud = False
        self._font = pygame.font.SysFont("consolas", 18)
        self._hud_tex = None
        self._hud_vao = None

        self.ctx.enable(moderngl.BLEND)
        try:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        except Exception:
            pass
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        try:
            self.ctx.multisample = True
        except Exception:
            pass

        self.prog = self.ctx.program(
            vertex_shader=_load("particle.vert"),
            fragment_shader=_load("particle.frag"),
        )
        self.glow_prog = self.ctx.program(
            vertex_shader=_load("quad.vert"),
            fragment_shader=_load("glow.frag"),
        )
        self.comp_prog = self.ctx.program(
            vertex_shader=_load("quad.vert"),
            fragment_shader=_load("composite.frag"),
        )
        self.blit_prog = self.ctx.program(
            vertex_shader=_load("quad.vert"),
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                in vec2 v_uv;
                out vec4 fragColor;
                void main() { fragColor = texture(u_tex, v_uv); }
            """,
        )
        self.hud_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    v_uv = in_uv;
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                in vec2 v_uv;
                out vec4 fragColor;
                void main() {
                    vec4 c = texture(u_tex, v_uv);
                    if (c.a < 0.05) discard;
                    fragColor = c;
                }
            """,
        )

        quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.quad_vao_glow = self.ctx.vertex_array(
            self.glow_prog, [(self.quad_vbo, "2f", "in_pos")]
        )
        self.quad_vao_comp = self.ctx.vertex_array(
            self.comp_prog, [(self.quad_vbo, "2f", "in_pos")]
        )
        self.quad_vao_blit = self.ctx.vertex_array(
            self.blit_prog, [(self.quad_vbo, "2f", "in_pos")]
        )

        self.vbo = self.ctx.buffer(particles.buffer.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "4f 4f 4f 4f", "in_a", "in_b", "in_c", "in_d")],
        )

        lut = build_heart_lut(S.HEART_LUT_SIZE)
        self.heart_tex = self.ctx.texture((S.HEART_LUT_SIZE, 1), 1, lut.tobytes(), dtype="f4")
        self.heart_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.heart_tex.repeat_x = True
        self.heart_tex.repeat_y = True

        self.scene_fbo = None
        self.bloom_a = None
        self.bloom_b = None
        self._ldr_tex = None
        self._ldr_fbo = None
        self._fbo_size = (0, 0)
        self.resize(S.TARGET_WIDTH, S.TARGET_HEIGHT)

    def resize(self, width: int = 0, height: int = 0) -> None:
        width, height = S.TARGET_WIDTH, S.TARGET_HEIGHT
        if (width, height) == self._fbo_size and self.scene_fbo is not None:
            return
        self._fbo_size = (width, height)
        bw, bh = max(width // 2, 32), max(height // 2, 32)

        def _tex(w, h):
            return self.ctx.texture((w, h), 4, dtype="f2")

        scene = _tex(width, height)
        ba = _tex(bw, bh)
        bb = _tex(bw, bh)
        for t in (scene, ba, bb):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[scene])
        self.bloom_a = self.ctx.framebuffer(color_attachments=[ba])
        self.bloom_b = self.ctx.framebuffer(color_attachments=[bb])
        self._ldr_tex = None
        self._ldr_fbo = None

    def upload_particles(self) -> None:
        packed = self.particles.pack()
        self.vbo.orphan(packed.nbytes)
        self.vbo.write(packed.tobytes())

    def render(self, galaxy, fps: float, win_size: tuple[int, int]) -> None:
        vw = self._fbo_size[0]
        vh = self._fbo_size[1]
        pulse = 0.5 + 0.5 * np.sin(galaxy.sim_time * 2.0 * np.pi * S.CORE_PULSE_HZ)

        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, vw, vh)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

        self.heart_tex.use(2)
        self.prog["u_heart"].value = 2
        self.prog["u_mvp"].write(self.camera.view_proj().astype("f4").tobytes())
        self.prog["u_time"].value = float(galaxy.sim_time)
        self.prog["u_form"].value = float(galaxy.formation_time)
        self.prog["u_color"].value = float(galaxy.color_intensity)
        self.prog["u_pulse"].value = float(pulse)
        self.prog["u_resolution"].value = (float(vw), float(vh))
        self.vao.render(moderngl.POINTS, vertices=self.particles.count)

        # Bloom: extract + horizontal, then vertical.
        self.bloom_a.use()
        self.ctx.viewport = (0, 0, self.bloom_a.width, self.bloom_a.height)
        self.ctx.disable(moderngl.BLEND)
        self.scene_fbo.color_attachments[0].use(0)
        self.glow_prog["u_tex"].value = 0
        self.glow_prog["u_dir"].value = (1.0, 0.0)
        self.glow_prog["u_threshold"].value = S.BLOOM_THRESHOLD
        self.glow_prog["u_mode"].value = 0
        self.quad_vao_glow.render(moderngl.TRIANGLE_STRIP)

        self.bloom_b.use()
        self.bloom_a.color_attachments[0].use(0)
        self.glow_prog["u_dir"].value = (0.0, 1.0)
        self.glow_prog["u_mode"].value = 1
        self.quad_vao_glow.render(moderngl.TRIANGLE_STRIP)

        # Extra blur pass for a softer halo.
        self.bloom_a.use()
        self.bloom_b.color_attachments[0].use(0)
        self.glow_prog["u_dir"].value = (1.0, 0.0)
        self.quad_vao_glow.render(moderngl.TRIANGLE_STRIP)
        self.bloom_b.use()
        self.bloom_a.color_attachments[0].use(0)
        self.glow_prog["u_dir"].value = (0.0, 1.0)
        self.quad_vao_glow.render(moderngl.TRIANGLE_STRIP)

        # Composite onto an offscreen target the size of the scene.
        self.scene_fbo.color_attachments[0].use(0)
        self.bloom_b.color_attachments[0].use(1)
        # We composite into bloom_a upscaled by drawing to screen with letterbox.
        # Use a dedicated full-res by drawing to default framebuffer.
        self.ctx.screen.use()
        win_w, win_h = win_size
        self.ctx.viewport = (0, 0, win_w, win_h)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        x, y, w, h = self.camera.letterbox(win_w, win_h)
        self.ctx.viewport = (x, y, w, h)
        self.comp_prog["u_scene"].value = 0
        self.comp_prog["u_bloom"].value = 1
        self.comp_prog["u_strength"].value = S.BLOOM_STRENGTH
        self.quad_vao_comp.render(moderngl.TRIANGLE_STRIP)

        if self.show_hud:
            self._draw_hud(galaxy, fps, win_w, win_h)

    def _draw_hud(self, galaxy, fps: float, win_w: int, win_h: int) -> None:
        lines = [
            f"HEART OF THE UNIVERSE   {fps:5.1f} fps   {self.particles.count:,} particles",
            f"speed {galaxy.speed:.2f}   color {galaxy.color_intensity:.2f}   zoom {self.camera.zoom:.2f}",
            "SPACE pause  R restart  UP/DOWN speed  C color  Z zoom  H hud  ESC quit",
        ]
        if galaxy.paused:
            lines[0] += "  [PAUSED]"
        text = "\n".join(lines)
        surf = self._font.render(text.replace("\n", "   |   "), True, (210, 225, 255))
        data = pygame.image.tobytes(surf, "RGBA", True)
        tw, th = surf.get_width(), surf.get_height()
        if self._hud_tex is None or self._hud_tex.size != (tw, th):
            self._hud_tex = self.ctx.texture((tw, th), 4, data)
            self._hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            self._hud_tex.write(data)

        # NDC quad in the top-left of the 9:16 letterbox.
        x, y, w, h = self.camera.letterbox(win_w, win_h)
        ndc_x0 = (x + 12) / win_w * 2.0 - 1.0
        ndc_y1 = 1.0 - (win_h - (y + h) + 12) / win_h * 2.0
        ndc_x1 = (x + 12 + tw) / win_w * 2.0 - 1.0
        ndc_y0 = 1.0 - (win_h - (y + h) + 12 + th) / win_h * 2.0
        verts = np.array(
            [
                ndc_x0, ndc_y0, 0.0, 0.0,
                ndc_x1, ndc_y0, 1.0, 0.0,
                ndc_x0, ndc_y1, 0.0, 1.0,
                ndc_x1, ndc_y1, 1.0, 1.0,
            ],
            dtype="f4",
        )
        vbo = self.ctx.buffer(verts.tobytes())
        vao = self.ctx.vertex_array(self.hud_prog, [(vbo, "2f 2f", "in_pos", "in_uv")])
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.viewport = (0, 0, win_w, win_h)
        self._hud_tex.use(0)
        self.hud_prog["u_tex"].value = 0
        vao.render(moderngl.TRIANGLE_STRIP)
        vao.release()
        vbo.release()
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

    def _ensure_ldr(self) -> None:
        w, h = S.TARGET_WIDTH, S.TARGET_HEIGHT
        if self._ldr_fbo is not None:
            return
        self._ldr_tex = self.ctx.texture((w, h), 3, dtype="f1")
        self._ldr_fbo = self.ctx.framebuffer(color_attachments=[self._ldr_tex])

    def compose_ldr(self) -> None:
        """Composite bloom into a cached 1080×1920 8-bit framebuffer."""
        self._ensure_ldr()
        w, h = S.TARGET_WIDTH, S.TARGET_HEIGHT
        self._ldr_fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.BLEND)
        self.scene_fbo.color_attachments[0].use(0)
        self.bloom_b.color_attachments[0].use(1)
        self.comp_prog["u_scene"].value = 0
        self.comp_prog["u_bloom"].value = 1
        self.comp_prog["u_strength"].value = S.BLOOM_STRENGTH
        self.quad_vao_comp.render(moderngl.TRIANGLE_STRIP)

    def read_frame_rgb(self) -> bytes:
        """GL bottom-left RGB24 bytes at 1080×1920 (vflip in ffmpeg)."""
        self.compose_ldr()
        return self._ldr_fbo.read(components=3, alignment=1)

    def capture_png(self, path: Path) -> None:
        """Save the composed scene at native 1080×1920."""
        w, h = S.TARGET_WIDTH, S.TARGET_HEIGHT
        data = self.read_frame_rgb()
        surf = pygame.image.frombytes(data, (w, h), "RGB")
        surf = pygame.transform.flip(surf, False, True)
        pygame.image.save(surf, str(path))

    def save_scene(self, path: Path) -> None:
        """Blit the HDR scene FBO to 8-bit and save (debug)."""
        w, h = self._fbo_size
        ldr = self.ctx.texture((w, h), 3, dtype="f1")
        fbo = self.ctx.framebuffer(color_attachments=[ldr])
        fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.BLEND)
        self.scene_fbo.color_attachments[0].use(0)
        self.blit_prog["u_tex"].value = 0
        self.quad_vao_blit.render(moderngl.TRIANGLE_STRIP)
        data = fbo.read(components=3)
        surf = pygame.image.frombytes(data, (w, h), "RGB")
        surf = pygame.transform.flip(surf, False, True)
        pygame.image.save(surf, str(path))
        fbo.release()
        ldr.release()
