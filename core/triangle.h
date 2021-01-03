#pragma once
#include "rz_types.h"
#include"color.h"

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
	Vertex() { hadColor = false; }
	Vertex(const Point3f& _p) { p = _p; hadColor = false; }
	// 把 Vertex 的位置与矩阵相乘
	void mul(const Matrix4x4f& m) {
		p = m * p;
	}

	Point3f p;
	float w;
	
	Color color;
	bool hadColor;
};

inline Log& operator<<(Log& log, const Vertex& v) {
	log << v.p;
	return log;
}

class Triangle {
public:
	Triangle(const Triple<Vertex*>& _v, const Triple<Point2f*>& _vt, const Triple<Vector3f*>& _vn) { v = _v; vt = _vt; vn = _vn; }
	~Triangle() {
		delete v[0];
		delete v[1];
		delete v[2];
		delete vt[0];
		delete vt[1];
		delete vt[2];
		delete vn[0];
		delete vn[1];
		delete vn[2];
		v[0] = v[1] = v[2] = nullptr;
		vt[0] = vt[1] = vt[2] = nullptr;
		vn[0] = vn[1] = vn[2] = nullptr;
	}

	Triple<Vertex*> v;
	Triple<Point2f*> vt;
	Triple<Vector3f*> vn;
};

inline Vector3f computeNormal(const Point3f& p0, const Point3f& p1, const Point3f& p2) {
	return unit_vector(cross(p1 - p0, p2 - p0));
}