#pragma once
#include <device_launch_parameters.h>
#include <vector>
#include "triangle.h"
#include "mesh.h"

class CudaMesh : public hitable {
public:
	__host__ __device__ CudaMesh() {}
 	__host__ __device__ CudaMesh(const Mesh* mesh) {
		// rebuild CudaMesh
		trianglelist = reinterpret_cast<Triangle*>(malloc(mesh->trianglelist.size()));
		trianglelist_size = mesh->trianglelist.size();
		vertex = reinterpret_cast<Vertex*>(malloc(mesh->vArray.size()));
		if (vertex == nullptr) return;
		vertex_size = mesh->vArray.size();
		vt = reinterpret_cast<Point2f*>(malloc(mesh->vtArray.size()));
		if (vt == nullptr) return;
		vt_size = mesh->vtArray.size();
		vn = reinterpret_cast<Vector3f*>(malloc(mesh->vnArray.size()));
		if (vn == nullptr) return;
		vn_size = mesh->vnArray.size();
		
		memcpy(trianglelist, mesh->trianglelist.data(), trianglelist_size);
		memcpy(vertex, mesh->vArray.data(), vertex_size);
		memcpy(vt, mesh->vtArray.data(), vertex_size);
		memcpy(vn, mesh->vnArray.data(), vertex_size);
	}
	__host__ __device__ ~CudaMesh() {}

	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record* rec) const override {
		hit_record temp_record;
		bool hit_anything = false;
		double closest_so_far = t_max;
		for (int index = 0; index < trianglelist_size; index++) {
			if (trianglelist[index].hit(r, t_min, closest_so_far, &temp_record)) {
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
		for (int index = 0; index < vertex_size; index++) {
			for (int axis = 0; axis < 3; axis++) {
				if (vertex[index].p[axis] < bl[axis]) {
					bl[axis] = vertex[index].p[axis];
				}
				if (vertex[index].p[axis] > ur[axis]) {
					ur[axis] = vertex[index].p[axis];
				}
			}
		}
		*box = aabb(Vector3f(bl[0], bl[1], bl[2]),
			Vector3f(ur[0], ur[1], ur[2]));
		return true;
	}

	Triangle* trianglelist;
	size_t trianglelist_size;
	Vertex* vertex;
	size_t vertex_size;
	Point2f* vt;
	size_t vt_size;
	Vector3f* vn;
	size_t vn_size;
};

