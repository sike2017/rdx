#pragma once
#include "rz_types.h"
#include"color.h"
#include "hitable.h"

inline void barycentric(const Point2f& q, const Point2f& p0, const Point2f& p1, const Point2f& p2, Point3f* out) {
	float gama = ((p0.y() - p1.y()) * q.x() + (p1.x() - p0.x()) * q.y() + p0.x() * p1.y() - p1.x() * p0.y()) /
		((p0.y() - p1.y()) * p2.x() + (p1.x() - p0.x()) * p2.y() + p0.x() * p1.y() - p1.x() * p0.y());
	float beta = ((p0.y() - p2.y()) * q.x() + (p2.x() - p0.x()) * q.y() + p0.x() * p2.y() - p2.x() * p0.y()) /
		((p0.y() - p2.y()) * p1.x() + (p2.x() - p0.x()) * p1.y() + p0.x() * p2.y() - p2.x() * p0.y());
	float alpha = 1.0f - beta - gama;

	*out = Vector3f(alpha, beta, gama);
}

template<typename T>
class Triple {
public:
	Triple<T>() {}
	Triple<T>(T v0, T v1, T v2) { e[0] = v0; e[1] = v1; e[2] = v2; }

	T& operator[](int index) { return e[index]; }
	const T operator[](int index) const { return e[index]; }
	T e[3];
};

class Vertex {
public:
	Vertex() {}
	Vertex(const Point3f& _p) { p = _p; }
	// 把 Vertex 的位置与矩阵相乘
	void mul(const Matrix4x4f& m) {
		p = m * p;
	}

	Point3f p;
	float w;
	
	Color color;
};

inline Log& operator<<(Log& log, const Vertex& v) {
	log << v.p;
	return log;
}

class Triangle : public hitable {
public:
	Triangle(const Triple<Vertex*>& _v, const Triple<Point2f*>& _vt, const Triple<Vector3f*>& _vn, material* mp) { v = _v; vt = _vt; vn = _vn; mat_ptr = mp; }
	~Triangle() {
		v[0] = v[1] = v[2] = nullptr;
		vt[0] = vt[1] = vt[2] = nullptr;
		vn[0] = vn[1] = vn[2] = nullptr;
	}

	virtual bool hit(const Ray& r, float t_min, float t_max, hit_record* rec) const override {
		Vector3f e0 = v[1]->p - v[0]->p;
		Vector3f e1 = v[2]->p - v[0]->p;

		Vector3f p = cross(r.direction(), e1);
		float det = dot(e0, p);
		if (det > -FLT_MIN && det < FLT_MIN) {
			return false;
		}
		float inv_det = 1.0f / det;

		Vector3f t_vector = r.origin() - v[0]->p;

		float u_float = dot(t_vector, p) * inv_det;
		if (u_float < 0.0f || u_float > 1.0f) {
			return false;
		}

		Vector3f q = cross(t_vector, e0);
		float v_float = dot(r.direction(), q) * inv_det;

		if (v_float < 0.0f || u_float + v_float > 1.0f) {
			return false;
		}

		float t_float = dot(e1, q) * inv_det;

		if (t_float < t_max && t_float > t_min) {
			rec->t = t_float;
			rec->p = r.point_at_parameter(t_float);
			rec->normal = unit_vector(cross(v[1]->p - v[0]->p, v[2]->p - v[0]->p));
			rec->mat_ptr = mat_ptr;
			Point3f barycentric_coordinate;
			barycentric(rec->p, v[0]->p, v[1]->p, v[2]->p, &barycentric_coordinate);
			float alpha = barycentric_coordinate[0];
			float beta = barycentric_coordinate[1];
			float gama = barycentric_coordinate[2];
			rec->u = alpha * vt[0]->x() + beta * vt[1]->x() + gama * vt[2]->x();
			rec->v = alpha * vt[0]->y() + beta * vt[1]->y() + gama * vt[2]->y();
			return true;
		}
		return false;
	}

	virtual bool bounding_box(float t0, float t1, aabb* box) const {
		float minx = std::min(v[0]->p.x(), std::min(v[1]->p.x(), v[2]->p.x()));
		float miny = std::min(v[0]->p.y(), std::min(v[1]->p.y(), v[2]->p.y()));
		float minz = std::min(v[0]->p.z(), std::min(v[1]->p.z(), v[2]->p.z()));
		float maxx = std::max(v[0]->p.x(), std::max(v[1]->p.x(), v[2]->p.x()));
		float maxy = std::max(v[0]->p.y(), std::max(v[1]->p.y(), v[2]->p.y()));
		float maxz = std::max(v[0]->p.z(), std::max(v[1]->p.z(), v[2]->p.z()));
		*box = aabb(Vector3f(minx, miny, minz),
			Vector3f(maxx, maxy, maxz));
		return true;
	}

	Triple<Vertex*> v;
	Triple<Point2f*> vt;
	Triple<Vector3f*> vn;
	material* mat_ptr;
};

inline Vector3f computeNormal(const Point3f& p0, const Point3f& p1, const Point3f& p2) { return unit_vector(cross(p1 - p0, p2 - p0)); }