#pragma once
#include <cuda_runtime.h>
#include <iostream>

#include "core/color.h"
#include "core/scenes.h"
#include "display/Painter.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

namespace ra { // rdx cuda
    __device__ RGBA to_rgba(const Color& color) {
        uint8_t ur = static_cast<int>(255.99 * range_float(color.r(), 0.0, 1.0));
        uint8_t ug = static_cast<int>(255.99 * range_float(color.g(), 0.0, 1.0));
        uint8_t ub = static_cast<int>(255.99 * range_float(color.b(), 0.0, 1.0));
        uint8_t ua = 255;
        uint32_t result;
        uint8_t* p = reinterpret_cast<uint8_t*>(&result);

        p[0] = ur;
        p[1] = ug;
        p[2] = ub;
        p[3] = ua;
    }

    __global__ void renderKernel(size_t width, size_t height, uint32_t* pixels, hitable* world) {
        int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int x = pixelIndex / width;
        int y = pixelIndex % width;
        if ((x >= width) || (y >= height)) return;
        int pixel_index = y * width + x;

        pixels[pixel_index] = to_rgba(Color(x / static_cast<float>(width), y / static_cast<float>(height), 0.2)).toRGBAUint32();
    }

	class RenderCuda {
    public:
        RenderCuda() {}
        ~RenderCuda() {}

        void render(size_t width, size_t height, uint32_t* pixels, hitable* world) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            maxThreadsPerBlock = prop.maxThreadsPerBlock;
            multiProcessorCount = prop.multiProcessorCount;

            renderKernel <<< multiProcessorCount, maxThreadsPerBlock >>> (width, height, pixels, world);
        }

        int maxThreadsPerBlock;
        int multiProcessorCount;

    private:
	};
}
