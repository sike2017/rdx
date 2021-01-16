#pragma once
#include "math/monolith_math.h"
#ifndef M_PI
#define M_PI 3.1415926
#endif

inline Matrix4x4f translation(float dx, float dy, float dz) {
	return Matrix4x4f(
		1, 0, 0, dx,
		0, 1, 0, dy,
		0, 0, 1, dz,
		0, 0, 0, 1
	);
}

inline Matrix4x4f rotateX(float theta) {
	theta = theta * M_PI / 180.0f;
	float sinth = sin(theta);
	float costh = cos(theta);
	return Matrix4x4f(
		1, 0, 0, 0,
		0, costh, -sinth, 0,
		0, sinth, costh, 0,
		0, 0, 0, 1
	);
}

inline Matrix4x4f rotateY(float theta) {
	theta = theta * M_PI / 180.0f;
	float sinth = sin(theta);
	float costh = cos(theta);
	return Matrix4x4f(
		costh, 0, sinth, 0,
		0, 1, 0, 0,
		-sinth, 0, costh, 0,
		0, 0, 0, 1
	);
}

inline Matrix4x4f rotateZ(float theta) {
	theta = theta * M_PI / 180.0f;
	float sinth = sin(theta);
	float costh = cos(theta);
	return Matrix4x4f(
		costh, -sinth, 0, 0,
		sinth, costh, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	);
}

inline Matrix4x4f scale(float sx, float sy, float sz) {
	return Matrix4x4f(
		sx, 0, 0, 0,
		0, sy, 0, 0,
		0, 0, sz, 0,
		0, 0, 0, 1
	);
}

inline Matrix4x4f unit_matrix() {
	return Matrix4x4f(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	);
}

class Transformer
{
public:
	virtual Matrix4x4f get_transform() const = 0;
};
