#version 450 core

layout (location = 0) out vec3 out_color;

void main() {
	const vec2 positions[] = {
		vec2(-0.5, -0.5),
		vec2(0.5, -0.5),
		vec2(0.0, 0.5)
	};

	const vec3 colors[] = {
		vec3(1.0, 0.0, 0.0),
		vec3(0.0, 1.0, 0.0),
		vec3(0.0, 0.0, 1.0),
	};
	
	out_color = colors[gl_VertexIndex];
	gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
