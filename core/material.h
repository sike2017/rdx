#pragma once
#include "hitable.h"
#include "rdx_random.h"
#include "texture.h"

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

class material {
public:
	__device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3f* attenuation, Ray* scattered) const { return false; }
	__device__ virtual Color emitted(float u, float v, const Vector3f& p) const {
		return Color(0, 0, 0);
	}

	float ns, ni, d, illum;
	Vector3f ka, kd, ks, ke;

	float rdx_light;
};

class lambertian : public material {
public:
	__device__ lambertian(rdxr_texture* a) : albedo(a) {}
	__device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3f* attenuation, Ray* scattered) const {
		Vector3f target = rec.p + rec.normal + random_in_unit_sphere();
		*scattered = Ray(rec.p, target - rec.p);
		*attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

	rdxr_texture* albedo;
};

class metal : public material {
public:
	__device__ metal(const Vector3f& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
	__device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3f* attenuation, Ray* scattered) const {
		Vector3f reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		*scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere());
		*attenuation = albedo;
		return (dot(scattered->direction(), rec.normal) > 0);
	}
	Vector3f albedo;
	float fuzz;
};

class dielectric : public material {
public:
	__device__ dielectric(float ri) : ref_idx(ri) {}
	__device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3f* attenuation, Ray* scattered) const {
		Vector3f outward_normal;
		Vector3f reflected = reflect(r_in.direction(), rec.normal);
		float ni_over_nt;
		*attenuation = Vector3f(1.0, 1.0, 1.0);
		Vector3f refracted;
		float reflect_prob;
		float cosine;
		if (dot(r_in.direction(), rec.normal) > 0) {
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
			cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}
		else {
			outward_normal = rec.normal;
			ni_over_nt = 1.0 / ref_idx;
			cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}
		if (refract(r_in.direction(), outward_normal, ni_over_nt, &refracted)) {
			reflect_prob = schlick(cosine, ref_idx);
		}
		else {
			*scattered = Ray(rec.p, reflected);
			reflect_prob = 1.0;
		}

		if (rdx_rand() < reflect_prob) {
			*scattered = Ray(rec.p, reflected);
		}
		else {
			*scattered = Ray(rec.p, refracted);
		}
		return true;
	}

	float ref_idx;
};

class diffuse_light : public material {
public:
	__device__ diffuse_light(rdxr_texture* a) : emit(a) {}
	__device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3f* attenuation, Ray* scattered) const { return false; }
	__device__ virtual Color emitted(float u, float v, const Vector3f& p) const override {
		return emit->value(u, v, p);
	}
	rdxr_texture* emit;
};
