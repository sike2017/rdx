#pragma once
#include <assert.h>
#include "transformation.h"
#include "triangle.h"
#include "material.h"

//template<typename T>
//class list_itor {
//public:
//	list_itor(T* _data, size_t _size) { data = _data; size = _size; }
//	~list_itor() {}
//
//	list_itor<T> get_index(size_t index) const {
//		list_itor<T> itor = *this;
//		itor.current_index = index;
//		return itor;
//	}
//
//	T operator*() const {
//		return data[current_index];
//	}
//
//	list_itor<T> operator++() {
//		assert(current_index < size);
//		return get_index(++current_index);
//	}
//
//	bool operator!=(const list_itor<T>& r) {
//		return current_index == r.current_index;
//	}
//
//	T* data;
//	size_t size;
//
//private:
//	size_t current_index = 0;
//};
//
//template<typename T>
//class base_list {
//public:
//	__host__ __device__ base_list() : p(nullptr), capacity(0), __size(0) {}
//	__host__ __device__ base_list(size_t s, T valu) {
//		this->capacity = s + 10;
//		this->__size = s;
//		this->p = new T[this->capacity];
//		// Because this coda may running on the GPU, so memset function might can not be used.
//		// We use assignment to set data one by one.
//		for (int index = 0; index < this->__size; index++) {
//			this->p[index] = valu;
//		}
//	}
//
//	__host__ __device__ ~base_list() {
//		if (this->p != nullptr) {
//			delete[] this->p;
//		}
//	}
//
//	__host__ __device__ base_list(const base_list& l) {
//		this->capacity = l.capacity;
//		this->__size = l.__size;
//		this->p = new T[this->capacity];
//		cudaMemcpy(this->p, l.p, this->__size, cudaMemcpyDefault); // cudaMemcpyDefault will chose the memcpy kind automaticully according to the address of pointer
//	}
//
//	__host__ __device__ void push_back(T data) {
//		if (this->p == nullptr) {
//			this->capacity = 10;
//			this->__size = 0;
//			this->p = new T[this->capacity];
//		}
//		if (this->capacity == this->__size) {
//			T* new_p = new T[2 * this->capacity];
//			cudaMemcpy(new_p, this->p, this->__size, cudaMemcpyDefault);
//			delete[] this->p;
//			this->p = new_p;
//		}
//		this->p[this->__size] = data;
//		this->__size++;
//	}
//
//	__host__ __device__ T pop_back() {
//		assert(__size >= 1);
//		if (__size >= 1) {
//			return p[__size--];
//		}
//		return NULL;
//	}
//
//	__host__ __device__ T& operator[](int index) {
//		assert(index >= 0 && index < __size);
//		return p[index];
//	}
//
//	__host__ __device__ T operator[](int index) const {
//		assert(index >= 0 && index < __size);
//		return p[index];
//	}
//
//	__host__ __device__ void operator=(const base_list & r) {
//		if (p != nullptr) {
//			delete[] p;
//		}
//		capacity = r.capacity;
//		__size = r.__size;
//
//		p = new T[capacity];
//		cudaMemcpy(p, r.p, __size, cudaMemcpyDefault);
//	}
//
//	__host__ __device__ bool empty() const {
//		return __size == 0;
//	}
//
//	__host__ __device__ size_t size() const {
//		return __size;
//	}
//
//	__host__ __device__ list_itor<T> begin() const {
//		auto itor = list_itor<T>(p, __size);
//		return itor.get_index(0);
//	}
//
//	__host__ __device__ list_itor<T> end() const {
//		auto itor = list_itor<T>(p, __size);
//		return itor.get_index(__size);
//	}
//
//	__host__ __device__ void add(T a) {
//		this->push_back(a);
//	}
//
//	T* p; // data
//	size_t capacity;
//	size_t __size;
//};

template<typename T>
class base_list : public std::vector<T> {
public:
	void add(T a) {
		this->push_back(a);
	}
};

template<typename T>
void createMaterial(const T& matr, material** d_matr) {
	material* mat_ptr;
	cudaMalloc(&mat_ptr, sizeof(T));
	cudaMemcpy(mat_ptr, &matr, sizeof(T), cudaMemcpyHostToDevice);
	*d_matr = mat_ptr;
}

class Mesh : public hitable {
public:
	enum MeshCopyDirectionKind {
		meshCopyHostToDevice,
		meshCopyDeviceToHost,
		meshCopyHostToHost,
		meshCopyDeviceToDevice
	};

	Mesh() {}
	~Mesh() {}

	void add(const Triangle& tg) {
		trianglelist.add(tg);
	}

	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record* rec) const override {
		hit_record temp_record;
		bool hit_anything = false;
		double closest_so_far = t_max;
		for (int index = 0; index < trianglelist.size(); index++) {
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
		for (int index = 0; index < vArray.size(); index++) {
			for (int axis = 0; axis < 3; axis++) {
				if (vArray[index].p[axis] < bl[axis]) {
					bl[axis] = vArray[index].p[axis];
				}
				if (vArray[index].p[axis] > ur[axis]) {
					ur[axis] = vArray[index].p[axis];
				}
			}
		}
		*box = aabb(Vector3f(bl[0], bl[1], bl[2]),
			Vector3f(ur[0], ur[1], ur[2]));
		return true;
	}

	void mul(const Matrix4x4f& m) {
		for (int index = 0; index < vArray.size(); index++) {
			vArray[index].p = m * vArray[index].p;
		}
	}

	void trans(float dx, float dy, float dz) {
		this->mul(transformation::translation(dx, dy, dz));
	}

	void scale(float sx, float sy, float sz) {
		this->mul(transformation::scale(sx, sy, sz));
	}

	base_list<Triangle> trianglelist;
	base_list<Vertex> vArray;
	base_list<Point2f> vtArray;
	base_list<Vector3f> vnArray;

};
typedef base_list<Mesh*> MeshList;