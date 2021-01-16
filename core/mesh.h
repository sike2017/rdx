#pragma once
#include <vector>
#include "rz_types.h"
#include "triangle.h"
#include "material.h"

template<typename T>
class base_list : public std::vector<T> {
public:
	void add(T a) {
		this->push_back(a);
	}
};

class Mesh : public hitable {
public:
	Mesh(material* mp) { mat_ptr = mp; }
	~Mesh() {
		for (Triangle* p : trianglelist) {
			delete p;
		}
	}
	
	void add(Triangle* tg) {
		trianglelist.add(tg);
	}

	void set_material(material* mp) {
		mat_ptr = mp;
	}

	material* get_material() const {
		return mat_ptr;
	}

	virtual bool hit(const Ray& r, float t_min, float t_max, hit_record* rec) const override {
		hit_record temp_record;
		bool hit_anything = false;
		double closest_so_far = t_max;
		for (Triangle* triangle : trianglelist) {
			if (triangle->hit(r, t_min, closest_so_far, &temp_record)) {
				hit_anything = true;
				closest_so_far = temp_record.t;
				*rec = temp_record;
			}
		}
		return hit_anything;
	}

	virtual bool bounding_box(float t0, float t1, aabb* box) const override {
		float bl[3] = { INFINITY, INFINITY, INFINITY };
		float ur[3] = { -INFINITY, -INFINITY, -INFINITY };
		for (Vertex* v : vArray) {
			for (int axis = 0; axis < 3; axis++) {
				if (v->p[axis] < bl[axis]) {
					bl[axis] = v->p[axis];
				}
				if (v->p[axis] > ur[axis]) {
					ur[axis] = v->p[axis];
				}
			}
		}
		*box = aabb(Vector3f(bl[0], bl[1], bl[2]), 
			Vector3f(ur[0], ur[1], ur[2]));
		return true;
	}

	void mul(const Matrix4x4f& m) {
		for (Vertex* v : vArray) {
			v->p = m * v->p;
		}
	}

	void trans(float dx, float dy, float dz) {
		this->mul(trans::translation(dx, dy, dz));
	}

	void scale(float sx, float sy, float sz) {
		this->mul(trans::scale(sx, sy, sz));
	}

	base_list<Triangle*> trianglelist;
	base_list<Vertex*> vArray;
	base_list<Point2f*> vtArray;
	base_list<Vector3f*> vnArray;

private:
	material* mat_ptr;
};

typedef base_list<Mesh*> MeshList;