#pragma once
#include "color.h"
#include "math/monolith_math.h"
#include "display/image_png.h"

class rdxr_texture {
public:
	__host__ __device__ virtual Color value(float u, float v, const Vector3f& p) const = 0;
};

class image_texture : public rdxr_texture {
public:
	__host__ __device__ image_texture(const std::string& image_file) {
		image = image_png::load_image(image_file.c_str());
	}
	__host__ __device__ ~image_texture() {
		image_png::free_image(&image);
	}
	__host__ __device__ virtual Color value(float u, float v, const Vector3f& p) const override {
		int i = (u)*image.width;
		int j = (1 - v) * image.height - 0.001;
		if (i < 0) i = 0;
		if (j < 0) j = 0;
		if (j > image.width - 1) j = image.width - 1;
		if (j > image.height - 1) j = image.height - 1;
		float r = int(image.p_buffer[3 * i + 3 * image.width * j]) / 255.0;
		float g = int(image.p_buffer[3 * i + 3 * image.height * j + 1]) / 255.0;
		float b = int(image.p_buffer[3 * i + 3 * image.height * j + 2]) / 255.0;
		return Color(r, g, b);
	}

	image_png::image_t image;
};

class solid_texture : public rdxr_texture {
public:
	__host__ __device__ solid_texture() {}
	__host__ __device__ solid_texture(const Color& c) : color(c) {}
	__host__ __device__ virtual Color value(float u, float v, const Vector3f& p) const {
		return color;
	}
	Color color;
};

class checker_texture : public rdxr_texture {
public:
	__host__ __device__ checker_texture() {}
	__host__ __device__ checker_texture(rdxr_texture* t0, rdxr_texture* t1): even(t0), odd(t1) {}
	__host__ __device__ virtual Color value(float u, float v, const Vector3f& p) const {
		float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
		if (sines < 0) {
			return odd->value(u, v, p);
		}
		else {
			return even->value(u, v, p);
		}
	}

	rdxr_texture* odd;
	rdxr_texture* even;
};