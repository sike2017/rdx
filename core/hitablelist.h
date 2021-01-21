#pragma once
#include "hitable.h"

class hitable_list : public hitable {
public:
	__device__ hitable_list() {}
	__device__ hitable_list(hitable** l, int n) { list = l; list_size = n; }
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, hit_record* rec) const override;
	virtual bool bounding_box(float t0, float t1, aabb* box) const override;
	hitable** list;
	int list_size;
};