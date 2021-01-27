#pragma once
#include <device_launch_parameters.h>
#include "ray.h"

class material;

struct hit_record {
	float t;
	Vector3f p;
	Vector3f normal;
	material* mat_ptr;
	float u;
	float v;
};

class aabb {
public:
	__host__ __device__ aabb() {}
	__host__ __device__ aabb(const Vector3f& a, const Vector3f& b) { _min = a; _max = b; }

	__host__ __device__ Vector3f min() const { return _min; }
	__host__ __device__ Vector3f max() const { return _max; }

	__device__ bool hit(const Ray& r, float tmin, float tmax) const {
		for (int a = 0; a < 3; a++) {
			float invD = 1.0f / r.direction()[a];
			float t0 = (min()[a] - r.origin()[a]) * invD;
			float t1 = (max()[a] - r.origin()[a]) * invD;
			if (invD < 0.0f) {
				std::swap(t0, t1);
			}
			tmin = t0 > tmin ? t0 : tmin;
			tmax = t1 > tmax ? t1 : tmax;
			if (tmax <= tmin) {
				return false;
			}
		}
		return true;
	}

	Vector3f _min;
	Vector3f _max;
};

class hitable {
public:
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record* rec) const = 0;
	virtual bool bounding_box(float t0, float t1, aabb* box) const = 0;
};

inline aabb surrounding_box(aabb box0, aabb box1) {
	Vector3f _small(util::min(box0.min().x(), box1.min().x()),
		util::min(box0.min().y(), box1.min().y()),
		util::min(box0.min().z(), box1.min().z()));
	Vector3f _big(util::max(box0.max().x(), box1.max().x()),
		util::max(box0.max().y(), box1.max().y()),
		util::max(box0.max().z(), box1.max().z()));
	return aabb(_small, _big);
}

