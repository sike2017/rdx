#pragma once

#include "math/monolith_math.h"

class Color : public Vector4f {
public:
	Color(float _r, float _g, float _b) { e[0] = _r; e[1] = _g; e[2] = _b; }
	Color(const Vector4f& v): Vector4f(v) {}
	Color() {}
	~Color() {}

	float r() const { return e[0]; }
	float g() const { return e[1]; }
	float b() const { return e[2]; }
};
