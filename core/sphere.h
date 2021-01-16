#pragma once
#include "hitable.h"

class sphere : public hitable {
public:
	sphere() {}
	sphere(Vector3f cen, float r, material* mp) : center(cen), radius(r), mat_ptr(mp) {}
	virtual bool hit(const Ray& r, float tmin, float tmax, hit_record* rec) const;
	virtual bool bounding_box(float t0, float t1, aabb* box) const;
	Vector3f center;
	float radius;
	material* mat_ptr;

private:
	void get_sphere_uv(const Vector3f& p, float* u, float* v) const {
		float phi = atan2(p.z(), p.x());
		float theta = asin(p.y());
		*u = 1 - (phi + M_PI) / (2 * M_PI);
		*v = (theta + M_PI / 2) / M_PI;
	}
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
			get_sphere_uv(rec->p, &rec->u, &rec->v);
			return true;
		}
		temp = (-b + sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec->t = temp;
			rec->p = r.point_at_parameter(rec->t);
			rec->normal = (rec->p - center) / radius;
			rec->mat_ptr = mat_ptr;
			get_sphere_uv(rec->p, &rec->u, &rec->v);
			return true;
		}
	}
	return false;
}

bool sphere::bounding_box(float t0, float t1, aabb* box) const {
	*box = aabb(center - Vector3f(radius, radius, radius), center + Vector3f(radius, radius, radius));
	return true;
}
