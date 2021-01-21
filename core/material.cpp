#include "material.h"

__device__ Vector3f reflect(const Vector3f& v, const Vector3f& n) {
	return v - 2 * dot(v, n) * n;
}

__device__ bool refract(const Vector3f& v, const Vector3f& n, float ni_over_nt, Vector3f* refracted) {
	Vector3f uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminant > 0) {
		*refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else {
		return false;
	}
}

__device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}
