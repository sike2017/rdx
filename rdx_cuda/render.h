#pragma once
#include <device_launch_parameters.h>
#include "core/color.h"
#include "core/hitable.h"
#include "core/rz_types.h"

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
