#include "hitable.h"

aabb surrounding_box(aabb box0, aabb box1) {
	Vector3f _small(std::min(box0.min().x(), box1.min().x()),
		std::min(box0.min().y(), box1.min().y()),
		std::min(box0.min().z(), box1.min().z()));
	Vector3f _big(std::max(box0.max().x(), box1.max().x()),
		std::max(box0.max().y(), box1.max().y()),
		std::max(box0.max().z(), box1.max().z()));
	return aabb(_small, _big);
}
