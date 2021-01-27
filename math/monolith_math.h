#pragma once
#include <device_launch_parameters.h>
#include "log/log.h"

class Vector4f
{
public:
	__host__ __device__ Vector4f(float _x = 0, float _y = 0, float _z = 0, float _w = 0) { e[0] = _x; e[1] = _y; e[2] = _z; e[3] = _w; }
	__host__ __device__ ~Vector4f() {}

	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }
	__host__ __device__ inline float w() const { return e[3]; }

	__host__ __device__ inline float& rx() { return e[0]; }
	__host__ __device__ inline float& ry() { return e[1]; }
	__host__ __device__ inline float& rz() { return e[2]; }
	__host__ __device__ inline float& rw() { return e[3]; }

	__host__ __device__ inline bool operator==(const Vector4f& r) const {
		return (e[0] == r.e[0] && e[1] == r.e[1] && e[2] == r.e[2] && e[3] == r.e[3]);
	}

	__host__ __device__ inline Vector4f operator+() const {
		return *this;
	}
	__host__ __device__ inline Vector4f operator-() const {
		return Vector4f(-e[0], -e[1], -e[2], -e[3]);
	}

	__host__ __device__ inline Vector4f operator+=(const Vector4f& r) {
		e[0] += r.e[0];
		e[1] += r.e[1];
		e[2] += r.e[2];
		e[3] += r.e[3];
		return *this;
	}
	__host__ __device__ inline Vector4f operator-=(const Vector4f& r) {
		e[0] -= r.e[0];
		e[1] -= r.e[1];
		e[2] -= r.e[2];
		e[3] -= r.e[3];
		return *this;
	}
	__host__ __device__ inline Vector4f operator*=(const Vector4f& r) {
		e[0] *= r.e[0];
		e[1] *= r.e[1];
		e[2] *= r.e[2];
		e[3] *= r.e[3];
		return *this;
	}
	__host__ __device__ inline Vector4f operator/=(const Vector4f& r) {
		e[0] /= r.e[0];
		e[1] /= r.e[1];
		e[2] /= r.e[2];
		e[3] /= r.e[3];
		return *this;
	}
	__host__ __device__ inline Vector4f operator*=(float k) {
		e[0] *= k;
		e[1] *= k;
		e[2] *= k;
		e[3] *= k;
		return *this;
	}
	__host__ __device__ inline Vector4f operator/=(float k) {
		e[0] /= k;
		e[1] /= k;
		e[2] /= k;
		e[3] /= k;
		return *this;
	}
	__host__ __device__ inline float& operator[](int index) {
		return e[index];
	}
	__host__ __device__ inline float operator[](int index) const {
		return e[index];
	}
	__host__ __device__ inline float operator()(int index) const {
		return e[index];
	}

	__host__ __device__ inline float length() const {
		return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	}
	__host__ __device__ float squared_length() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	float e[4];
};

__host__ __device__ inline Vector4f operator+(const Vector4f& l, const Vector4f& r) {
	return Vector4f(l.x() + r.x(), l.y() + r.y(), l.z() + r.z(), l.w() + r.w());
}
__host__ __device__ inline Vector4f operator-(const Vector4f& l, const Vector4f& r) {
	return Vector4f(l.x() - r.x(), l.y() - r.y(), l.z() - r.z(), l.w() - r.w());
}
__host__ __device__ inline Vector4f operator*(const Vector4f& l, const Vector4f& r) {
	return Vector4f(l.x() * r.x(), l.y() * r.y(), l.z() * r.z(), l.w() * r.w());
}
__host__ __device__ inline Vector4f operator/(const Vector4f& l, const Vector4f& r) {
	return Vector4f(l.x() / r.x(), l.y() / r.y(), l.z() / r.z(), l.w() / r.w());
}
__host__ __device__ inline Vector4f operator*(const Vector4f& l, float r) {
	return Vector4f(l.x() * r, l.y() * r, l.z() * r, l.w() * r);
}
__host__ __device__ inline Vector4f operator/(const Vector4f& l, float r) {
	return Vector4f(l.x() / r, l.y() / r, l.z() / r, l.w() / r);
}

__host__ __device__ inline Vector4f operator*(float left, const Vector4f& right) {
	return Vector4f(left * right.e[0], left * right.e[1], left * right.e[2], left * right.e[3]);
}

typedef Vector4f Vector3f;
typedef Vector4f Vector2f;

__host__ __device__ inline Vector4f unit_vector(const Vector3f& v) {
	Vector4f ret = v;
	float k = 1.0 / sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
	ret.e[0] *= k; ret.e[1] *= k; ret.e[2] *= k;
	return ret;
}

__host__ __device__ inline float dot(const Vector3f& v0, const Vector3f& v1) {
	return v0.e[0] * v1.e[0] + v0.e[1] * v1.e[1] + v0.e[2] * v1.e[2];
}

__host__ __device__ inline Vector3f cross(const Vector3f& v0, const Vector3f& v1) {
	return Vector3f((v0.e[1] * v1.e[2] - v0.e[2] * v1.e[1]),
		(-(v0.e[0] * v1.e[2] - v0.e[2] * v1.e[0])),
		(v0.e[0] * v1.e[1] - v0.e[1] * v1.e[0]));
}

class Matrix4x1f {
public:
	__host__ __device__ Matrix4x1f() {
		memset(e, 0, 4 * sizeof(float));
	}
	__host__ __device__ Matrix4x1f(float x, float y, float z, float w) { e[0] = x; e[1] = y; e[2] = z; e[3] = w; }
	__host__ __device__ Matrix4x1f(const Vector4f& v) { e[0] = v.x(); e[1] = v.y(); e[2] = v.z(); e[3] = v.w(); }
	__host__ __device__ ~Matrix4x1f() {}

	__host__ __device__ operator Vector4f() { return Vector4f(e[0], e[1], e[2], e[3]); }
	__host__ __device__ float& operator[](int index) { return e[index]; }
	__host__ __device__ float& operator()(int row, int column) { return e[row * 1 + column]; }
	__host__ __device__ float get(int row, int column) const { return e[row * 1 + column]; }

	float e[4];
};

class Matrix4x4f {
public:
	__host__ __device__ Matrix4x4f() {
		memset(e, 0, 16 * sizeof(float));
	}
	__host__ __device__ Matrix4x4f(float e00, float e01, float e02, float e03,
		float e10, float e11, float e12, float e13,
		float e20, float e21, float e22, float e23,
		float e30, float e31, float e32, float e33) {
		e[0] = e00; e[1] = e01; e[2] = e02; e[3] = e03;
		e[4] = e10; e[5] = e11; e[6] = e12; e[7] = e13;
		e[8] = e20; e[9] = e21; e[10] = e22; e[11] = e23;
		e[12] = e30; e[13] = e31; e[14] = e32; e[15] = e33;
	}
	__host__ __device__ ~Matrix4x4f() {}

	__host__ __device__ float& operator[](int index) { return e[index]; }
	__host__ __device__ float& operator()(int row, int column) { return e[row * 4 + column]; }
	__host__ __device__ float get(int row, int column) const { return e[row * 4 + column]; }

	__host__ __device__ Matrix4x4f operator*(Matrix4x4f mat) const {
		Matrix4x4f result;
		float temp;
		int row = 4;
		int column = 4;
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < column; j++)
			{
				temp = 0;
				for (int a = 0; a < column; a++)
				{
					temp += (get(i, a) * mat.get(a, j));
				}
				result(i, j) = temp;
			}
		}

		return result;
	}
	__host__ __device__ Matrix4x1f operator*(Matrix4x1f mat) const {
		Matrix4x1f result;
		float temp;
		int row = 4;
		int column = 1;
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < column; j++)
			{
				temp = 0;
				for (int a = 0; a < 4; a++)
				{
					temp += (get(i, a) * mat.get(a, j));
				}
				result(i, j) = temp;
			}
		}

		return result;
	}
	__host__ __device__ Matrix4x4f operator*=(Matrix4x4f mat) {
		float temp;
		int row = 4;
		int column = 4;
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < column; j++)
			{
				temp = 0;
				for (int a = 0; a < column; a++)
				{
					temp += (get(i, a) * mat.get(a, j));
				}
				(*this)(i, j) = temp;
			}
		}

		return *this;
	}

	float e[16];
};

__host__ __device__ inline Log operator<<(Log& log, const Vector4f& v) {
	log << "[" << v.x() << ", " << v.y() << ", " << v.z() << ", " << v.w() << "]" << rendl;
	return log;
}

__host__ __device__ inline Log operator<<(Log& log, const Matrix4x4f& m) {
	log << "[";
	for (int row = 0; row < 4; row++) {
		log << "[";
		for (int column = 0; column < 4; column++) {
			log << m.get(row, column) << (column != 3 ? ", " : "");
		}
		log << "]" << (row != 3 ? ",\n" : "");
	}
	log << "]\n";
	return log;
}

__host__ __device__ inline Log operator<<(Log& log, const Matrix4x1f& m) {
	log << "[";
	for (int row = 0; row < 4; row++) {
		log << "[";
		for (int column = 0; column < 1; column++) {
			log << m.get(row, column) << (column != 0 ? ", " : "");
		}
		log << "]" << (row != 3 ? ",\n" : "");
	}
	log << "]\n";
	return log;
}

namespace util {
	template<typename T>
	void swap(T& a, T& b) {
		T tmp = a;
		a = b;
		b = tmp;
	}
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
	template<typename T>
	__host__ __device__ T min(T a, T b) {
		return a < b ? a : b;
	}
	template<typename T>
	__host__ __device__ T max(T a, T b) {
		return a > b ? a : b;
	}

	bool solveQuadratic(const float& a, const float& b, const float& c, float* x0, float* x1);
}

#ifndef M_PI
#define M_PI 3.1415926
#endif