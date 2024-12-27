#version 450 core

layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec2 a_uv;

layout (location = 0) out vec2 out_uv;

layout (set = 0, binding = 0) uniform View {
	mat4 mat_vp;
};

void main() {
	out_uv = a_uv;
	const vec4 v_pos = vec4(a_pos, 0.0, 1.0);
	gl_Position = mat_vp * v_pos;
}
