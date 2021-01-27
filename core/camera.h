#pragma once
#include "ray.h"
#include "rdx_rand.h"

class camera {
public:
	__host__ __device__ camera() {}
 	__host__ __device__ camera(const Vector3f& lookfrom, const Vector3f& lookat, const Vector3f& vup, float vfov, float aspect, float aperture, float focus_dist) {
		lens_radius = aperture / 2;
		float theta = vfov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
		horizontal = 2 * half_width * focus_dist * u;
		vertical = 2 * half_height * focus_dist * v;
	}
	__device__ Ray get_ray(float s, float t, curandState* state) const {
		Vector3f rd = lens_radius * random_in_unit_disk(state);
		Vector3f offset = u * rd.x() + v * rd.y();
		return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}

	Vector3f origin;
	Vector3f lower_left_corner;
	Vector3f horizontal;
	Vector3f vertical;
	Vector3f u, v, w;
	float lens_radius;

private:
	__device__ Vector3f random_in_unit_disk(curandState* state) const {
		Vector3f p;
		do {
			p = 2.0 * Vector3f(device_rand(state), device_rand(state), 0) - Vector3f(1, 1, 0);
		} while (dot(p, p) >= 1.0);
		return p;
	}
};