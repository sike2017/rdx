#pragma once
#include <cuda_runtime.h>
#include "core/color.h"
#include "core/hitablelist.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace ra { // rdx cuda
    __device__ uint32_t to_u(const Color& color);

    __global__ void renderKernel(size_t width, size_t height, uint32_t* pixels, hitable* scene);

    class RenderCuda {
    public:
        RenderCuda() {}
        ~RenderCuda() {}
        void render(size_t width, size_t height, uint32_t* host_pixels);
        int maxThreadsPerBlock;
        int multiProcessorCount;

    private:
    };
}
