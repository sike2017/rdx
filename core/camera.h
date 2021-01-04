#pragma once
#include "ray.h"

class camera {
public:
	camera(const Vector3f& lookfrom, const Vector3f& lookat, const Vector3f& vup, float vfov, float aspect) {
		float theta = vfov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * u - half_height * v - w;
		horizontal = 2 * half_width * u;
		vertical = 2 * half_height * v;
	}
	Ray get_ray(float s, float t) const { return Ray(origin, lower_left_corner + s * horizontal + t * vertical - origin); }

	Vector3f origin;
	Vector3f lower_left_corner;
	Vector3f horizontal;
	Vector3f vertical;
	Vector3f u, v, w;
	float lens_radius;
};
