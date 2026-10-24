#version 330 core

uniform sampler2D u_tex;
uniform vec2 u_dir;
uniform float u_threshold;
uniform int u_mode; // 0 bright extract+blur, 1 blur only

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_tex, 0));
    vec3 acc = vec3(0.0);
    const float w0 = 0.227027;
    const float w1 = 0.1945946;
    const float w2 = 0.1216216;
    const float w3 = 0.054054;
    const float w4 = 0.016216;
    float weights[5] = float[](w0, w1, w2, w3, w4);

    for (int i = -4; i <= 4; ++i) {
        vec2 uv = v_uv + u_dir * texel * float(i) * 1.7;
        vec3 s = texture(u_tex, uv).rgb;
        if (u_mode == 0) {
            float lum = dot(s, vec3(0.299, 0.587, 0.114));
            s *= smoothstep(u_threshold, u_threshold + 0.55, lum);
        }
        acc += s * weights[abs(i)];
    }
    fragColor = vec4(acc, 1.0);
}
