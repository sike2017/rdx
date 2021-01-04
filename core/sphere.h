#pragma once
#include "hitable.h"

class sphere : public hitable {
public:
	sphere() {}
	sphere(Vector3f cen, float r, material* mp) : center(cen), radius(r), mat_ptr(mp) {};
	virtual bool hit(const Ray& r, float tmin, float tmax, hit_record* rec) const;
	Vector3f center;
	float radius;
	material* mat_ptr;
};

bool sphere::hit(const Ray& r, float t_min, float t_max, hit_record* rec) const {
	Vector3f oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec->t = temp;
			rec->p = r.point_at_parameter(rec->t);
			rec->normal = (rec->p - center) / radius;
			rec->mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec->t = temp;
			rec->p = r.point_at_parameter(rec->t);
			rec->normal = (rec->p - center) / radius;
			rec->mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}
