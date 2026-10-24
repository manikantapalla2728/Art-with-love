#version 330 core

layout(location = 0) in vec4 in_a; // theta/x, radius/y, z, type
layout(location = 1) in vec4 in_b; // rgb, size
layout(location = 2) in vec4 in_c; // brightness, alpha, delay, omega
layout(location = 3) in vec4 in_d; // phase, twinkle, seed, spread

uniform mat4 u_mvp;
uniform float u_time;
uniform float u_form;
uniform float u_color;
uniform float u_pulse;
uniform vec2 u_resolution;
uniform sampler2D u_heart;

out vec3 v_color;
out float v_alpha;
out float v_type;
out float v_bright;

const float PI = 3.141592653589793;
const float TWO_PI = 6.283185307179586;
const float CORE_R = 0.13;

float heartR(float theta) {
    float u = fract(theta / TWO_PI + 0.5);
    return texture(u_heart, vec2(u, 0.5)).r;
}

void main() {
    float a0 = in_a.x;
    float b0 = in_a.y;
    float z0 = in_a.z;
    float ptype = in_a.w;

    vec3 rgb = in_b.xyz;
    float psize = in_b.w;
    float bright = in_c.x;
    float alpha = in_c.y;
    float delay = in_c.z;
    float omega = in_c.w;
    float phase = in_d.x;
    float twinkle = in_d.y;
    float seed = in_d.z;

    float appear = smoothstep(delay, delay + 0.9, u_form);
    appear *= appear;

    vec3 pos;
    if (ptype < 0.5) {
        float drift = u_time * 0.01;
        pos = vec3(
            a0 + sin(seed * 6.1 + u_time * 0.04) * 0.012,
            b0 + drift * (seed - 0.5) * 0.12,
            z0
        );
        appear = smoothstep(delay, delay + 0.35, u_form);
    } else {
        float grow = mix(0.018, 1.0, clamp((u_form - delay) / 2.6, 0.0, 1.0));
        if (ptype > 2.5 && ptype < 3.5) {
            grow = mix(0.12, 1.0, clamp((u_form - delay) / 1.15, 0.0, 1.0));
        }
        float r = b0 * grow;
        float theta = a0 + omega * u_time;
        theta += 0.12 * sin(r * 8.5 + seed * 14.0 + u_time * 0.18);

        float Rh = heartR(theta);
        float blend = clamp((r - 0.16) / 0.50, 0.0, 1.0);
        blend = pow(blend, 0.82);
        float R = mix(0.70, Rh, blend);

        float n1 = sin(seed * 17.2 + theta * 2.0 + u_time * 0.28);
        float n2 = cos(seed * 9.4 + theta * 3.1 - u_time * 0.21);
        float turbAmp = (0.006 + 0.012 * r) * (ptype < 1.5 ? 1.55 : 0.85);

        pos.x = r * R * cos(theta) + n1 * turbAmp;
        pos.y = r * R * sin(theta) + n2 * turbAmp;
        pos.z = z0 * (1.0 - 0.4 * r) + 0.006 * n1;

        if (ptype > 2.5 && ptype < 3.5) {
            pos.xy *= 1.0 + u_pulse * 0.035;
        }
    }

    gl_Position = u_mvp * vec4(pos, 1.0);

    float tw = 0.74 + 0.26 * sin(u_time * twinkle + phase);
    if (ptype > 3.5) {
        tw = 0.5 + 0.5 * pow(0.5 + 0.5 * sin(u_time * twinkle * 1.35 + phase), 3.0);
    }

    float depth = 0.78 + 0.22 * (0.5 - pos.z);
    v_color = rgb * u_color * depth;
    v_alpha = alpha * appear * tw;
    v_type = ptype;
    v_bright = bright * appear;

    float resScale = u_resolution.y / 1920.0;
    float px = psize;
    if (ptype > 2.5 && ptype < 3.5) {
        px *= 1.0 + u_pulse * 0.06;
    }
    gl_PointSize = clamp(px * resScale * (0.9 + 0.25 * depth), 1.0, 170.0);
}
