#version 450 core

layout (location = 0) in vec4 in_tint;
layout (location = 1) in vec2 in_uv;

layout (location = 0) out vec4 out_color;

void main() {
	out_color = in_tint;
}
