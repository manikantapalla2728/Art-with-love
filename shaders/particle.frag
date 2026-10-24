#version 330 core

in vec3 v_color;
in float v_alpha;
in float v_type;
in float v_bright;

out vec4 fragColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    float core = exp(-r2 * 10.5);
    float halo = exp(-r2 * 2.6);
    float glow = exp(-r2 * 0.95);

    float spike = 0.0;
    if (v_type > 3.5) {
        float ax = exp(-abs(p.x) * 22.0) * (1.0 - abs(p.y));
        float ay = exp(-abs(p.y) * 22.0) * (1.0 - abs(p.x));
        float d1 = abs(p.x - p.y);
        float d2 = abs(p.x + p.y);
        float ad = exp(-d1 * 26.0) * exp(-length(p) * 1.8);
        float bd = exp(-d2 * 26.0) * exp(-length(p) * 1.8);
        spike = (ax + ay) * 1.05 + (ad + bd) * 0.40;
        core *= 1.35;
    }

    float nebula = 0.0;
    if (v_type > 0.5 && v_type < 1.5) {
        nebula = exp(-r2 * 1.05) * 0.62;
        core *= 0.18;
        halo *= 0.70;
        glow *= 1.15;
    }

    if (v_type > 2.5 && v_type < 3.5) {
        core = exp(-r2 * 14.0);
        halo = exp(-r2 * 3.6);
        glow = exp(-r2 * 1.4) * 0.55;
    }

    float shape = core * 1.15 + halo * 0.38 + glow * 0.10 + spike + nebula;
    float intensity = shape * v_alpha * v_bright;
    vec3 col = v_color * intensity;
    fragColor = vec4(col, intensity);
}
