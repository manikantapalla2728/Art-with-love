#version 330 core

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_strength;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;
    vec3 color = scene + bloom * u_strength;

    float lum = max(dot(color, vec3(0.2126, 0.7152, 0.0722)), 1e-5);
    float mapped = 1.0 - exp(-lum * 0.78);
    color *= mapped / lum;

    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luma), color, 1.22);

    color = pow(clamp(color, 0.0, 1.0), vec3(0.90));
    fragColor = vec4(color, 1.0);
}
