#pragma once
#include "ray.h"

class camera {
public:
	camera() {
		lower_left_corner = Vector3f(-2.0, -1.0, -1.0);
		horizontal = Vector3f(4.0, 0.0, 0.0);
		vertical = Vector3f(0.0, 2.0, 0.0);
		origin = Vector3f(0.0, 0.0, 0.0);
	}
	Ray get_ray(float u, float v) { return Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin); }

	Vector3f origin;
	Vector3f lower_left_corner;
	Vector3f horizontal;
	Vector3f vertical;
};
