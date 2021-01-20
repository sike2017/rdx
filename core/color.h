#pragma once
#include <device_launch_parameters.h>
#include "math/monolith_math.h"

class Color : public Vector4f {
public:
	__device__ Color(float _r, float _g, float _b) { e[0] = _r; e[1] = _g; e[2] = _b; }
	__device__ Color(const Vector4f& v) : Vector4f(v) {}
	__device__ Color() {}
	__device__ ~Color() {}

	__device__ float r() const { return e[0]; }
	__device__ float g() const { return e[1]; }
	__device__ float b() const { return e[2]; }
};
inline float range_float(float f, float fmin, float fmax) {
	return f < fmin ? fmin: f > fmax ? fmax : f;
}
