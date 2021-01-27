#pragma once
#include <iostream>
#include <driver_types.h>

#include "math/monolith_math.h"
#include "texture.h"
#include "hitable.h"

typedef Vector3f Point3f;
typedef Vector2f Point2f;

#define DRAW_PRIMITIVE_SUBSTANCE_VERTEX 1
#define DRAW_PRIMITIVE_SUBSTANCE_LINE 2
#define DRAW_PRIMITIVE_SUBSTANCE_FACE 4

typedef int DRAW_PRIMITIVE_SUBSTANCE_STATUS;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

template<typename T>
class Triple {
public:
	__host__ __device__ Triple<T>() {}
	__host__ __device__ Triple<T>(T v0, T v1, T v2) { e[0] = v0; e[1] = v1; e[2] = v2; }

	__host__ __device__ T& operator[](int index) { return e[index]; }
	__host__ __device__ const T operator[](int index) const { return e[index]; }
	T e[3];
};

class Vertex {
public:
	__host__ __device__ Vertex() {}
	__host__ __device__ Vertex(const Point3f& _p) { p = _p; }
	// 把 Vertex 的位置与矩阵相乘
	__host__ __device__ void mul(const Matrix4x4f& m) {
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

template<class T>
class AssetNode {
public:
	AssetNode() { asset_host = nullptr; asset_device = nullptr; }
	void clear() {
		if (asset_host != nullptr) {
			delete asset_host;
		}
		if (asset_device != nullptr) {
			cudaFree(asset_device);
		}
	}

	AssetNode<rdxr_texture> to_texture() {
		AssetNode<rdxr_texture> node;
		node.asset_host = this->asset_host;
		node.asset_device = this->asset_device;
		return node;
	}
	AssetNode<material> to_material() {
		AssetNode<material> node;
		node.asset_host = this->asset_host;
		node.asset_device = this->asset_device;
		return node;
	}

	T* asset_host;
	T* asset_device;
};
