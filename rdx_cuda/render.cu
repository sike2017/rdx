#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "render.h"
#include "core/color.h"
#include "core/scenes.h"
#include "core/camera.h"
#include "core/rz_types.h"
#include "core/sphere.h"

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
    __device__ uint32_t to_u(const Color& color) {
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
        return result;
    }

    __device__ Color color(const Ray& r, hitable** scene) {
        Ray cur_ray = r;
        Color emitted;
        Ray scatter_ray;
        Color attenuation;
        Color col(1, 1, 1);
        for (int i = 0; i < 50; i++) {
            hit_record rec;
            if ((*scene)->hit(cur_ray, 0.001f, FLT_MAX, &rec)) {
                emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
                if (rec.mat_ptr->scatter(cur_ray, rec, &attenuation, &scatter_ray)) {
                    col = col * (emitted + attenuation * col);
                    cur_ray = scatter_ray;
                }
                else {
                    return emitted;
                }
            }
            else {
                Vector3f unit_direction = unit_vector(r.direction());
                float t = 0.5 * (unit_direction.y() + 1.0);
                Color c = (1.0 - t) * Vector3f(1.0, 1.0, 1.0) + t * Vector3f(0.5, 0.7, 1.0);
                return col * c;
            }
        }
    }

    __global__ void renderKernel(size_t width, size_t height, const camera cam, uint32_t* d_pixels, hitable* scene, size_t totalThreads, size_t ns) {
        for (int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x; pixelIndex < width * height; pixelIndex += totalThreads) {
            int x = pixelIndex % width;
            int y = pixelIndex / width;
            d_pixels[pixelIndex] = to_u(Color(x / static_cast<float>(width), y / static_cast<float>(height), 0.2));
            //Color col(0, 0, 0);
            //for (int s = 0; s < ns; s++) {
            //    float u = static_cast<float>(x + rdx_rand()) / static_cast<float>(width);
            //    float v = static_cast<float>(y + rdx_rand()) / static_cast<float>(height);
            //    Ray r = cam.get_ray(u, v);
            //    col += color(r, &scene);
            //}
            //col /= static_cast<float>(ns);
            //col = Color(sqrt(col.r()), sqrt(col.g()), sqrt(col.b()));
            //d_pixels[pixelIndex] = to_u(col);
            //printf("pixelIndex: %d\n", pixelIndex);
        }
    }

    void RenderCuda::render(size_t width, size_t height, uint32_t* host_pixels) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        maxThreadsPerBlock = prop.maxThreadsPerBlock;
        multiProcessorCount = prop.multiProcessorCount;
        uint32_t* d_pixels;
        cudaMalloc(&d_pixels, sizeof(uint32_t) * width * height);
        size_t ns = 100;
        sphere* d_sphere;
        cudaMalloc(&d_sphere, sizeof(sphere));
        renderKernel<<<multiProcessorCount, maxThreadsPerBlock>>>(width, height, camera(), d_pixels, d_sphere, maxThreadsPerBlock * multiProcessorCount, ns);
        printf("kernel done\n");
        cudaMemcpy(host_pixels, d_pixels, 4 * width * height, cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

