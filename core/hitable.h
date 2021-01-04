#pragma once
#include "ray.h"

class material;

struct hit_record {
	float t;
	Vector3f p;
	Vector3f normal;
	material* mat_ptr;
};

class hitable {
public:
	virtual bool hit(const Ray& r, float t_min, float t_max, hit_record* rec) const = 0;
};
