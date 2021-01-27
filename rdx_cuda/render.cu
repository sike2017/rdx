#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include "render.h"
#include "core/color.h"
#include "core/camera.h"
#include "core/rz_types.h"
#include "core/sphere.h"
#include "core/hitablelist.h"
#include "core/scenes.h"

__global__ void createScene(hitable** d_hitable, camera** d_cam, size_t width, size_t height, curandState* state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Point3f lookfrom(13, 2, 3);
        Point3f lookat(0, 0, 0);
        int nums = 2;
        hitable** list = new hitable * [nums];
        list[0] = new sphere(Point3f(0, 0, 0), 1, new metal((0.2, 0.4, 0.1), 0));
        list[1] = new sphere(Point3f(0, -1001, 0), 1000, new lambertian(new checker_texture(new solid_texture(Color(0.12, 0.19, 0.25)), new solid_texture(Color(0.9, 0.9, 0.9)))));
        *d_hitable = new hitable_list(list, nums);
        *d_hitable = random_scene(state);
        *d_cam = new camera(lookfrom, lookat, Vector3f(0, 1, 0), 40.0, static_cast<float>(width) / height, 0.1, (lookfrom - lookat).length());
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

    __device__ Color color(const Ray& r, hitable** scene, curandState* state) {
        Ray cur_ray = r;
        Color emitted;
        Ray scatter_ray;
        Color attenuation;
        Color cur_attenuation(1, 1, 1);
        for (int i = 0; i < 50; i++) {
            hit_record rec;
            if ((*scene)->hit(cur_ray, 0.001f, FLT_MAX, &rec)) {
                emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
                if (rec.mat_ptr->scatter(cur_ray, rec, &attenuation, &scatter_ray, state)) {
                    cur_attenuation = cur_attenuation * (emitted + attenuation * cur_attenuation);
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
                return cur_attenuation * c;
            }
        }
        
        return Color(0, 0, 0);
    }

    __global__ void renderKernel(size_t width, size_t height, camera** d_cam, uint32_t* d_pixels, hitable** scene, size_t totalThreads, size_t ns, curandState* rand_state, size_t* d_thread_pixel_nums) {
        int renderIndex = blockIdx.x * blockDim.x + threadIdx.x;
        d_thread_pixel_nums[renderIndex] = 0;
        for (int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x; pixelIndex < width * height; pixelIndex += totalThreads) {
            int x = pixelIndex % width;
            int y = pixelIndex / width;
            d_pixels[pixelIndex] = to_u(Color(x / static_cast<float>(width), y / static_cast<float>(height), 0.2));
            Color col(0, 0, 0);
            curandState* localState = &rand_state[pixelIndex];
            for (int s = 0; s < ns; s++) {
                float u = static_cast<float>(x + device_rand(localState)) / static_cast<float>(width);
                float v = static_cast<float>(y + device_rand(localState)) / static_cast<float>(height);
                Ray r = (*d_cam)->get_ray(u, v, localState);
                col += color(r, scene, localState);
            }
            col /= static_cast<float>(ns);
            col = Color(sqrt(col.r()), sqrt(col.g()), sqrt(col.b()));
            d_pixels[pixelIndex] = to_u(col);
            d_thread_pixel_nums[renderIndex]++;
        }
    }

    __global__ void renderInit(size_t width, size_t height, size_t totalThreads, curandState* rand_state) {
        int renderIndex = blockIdx.x * blockDim.x + threadIdx.x;
        for (int pixelIndex = renderIndex; pixelIndex < width * height; pixelIndex += totalThreads) {
            curand_init(1942, pixelIndex, 0, &rand_state[pixelIndex]);
        }
    }

    __global__ void pixelCount(size_t* d_pixel_nums, size_t totalThreads, size_t totalPixels, size_t* d_pixel_count) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            size_t pixel_sums = 0;
            for (int index = 0; index < totalThreads; index++) {
                pixel_sums += (d_pixel_nums[index]);
            }
            *d_pixel_count = pixel_sums;
        }
        else {
            printf("the pixelCount function only can be called on 1 thread\n");
        }
    }

    void RenderCuda::render(size_t width, size_t height, uint32_t* host_pixels) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        maxThreadsPerBlock = prop.maxThreadsPerBlock;
        multiProcessorCount = prop.multiProcessorCount;
        uint32_t* d_pixels;
        checkCudaErrors(cudaMalloc(&d_pixels, sizeof(uint32_t) * width * height));
        size_t ns = 100;
        size_t blocks = 3000;
        size_t threads = 100;
        curandState* d_rand_state;
        checkCudaErrors(cudaMalloc(&d_rand_state, sizeof(curandState) * width * height));
        renderInit<<<blocks, threads>>>(width, height, threads * blocks, d_rand_state);
        checkCudaErrors(cudaDeviceSynchronize());
        hitable** d_scene;
        checkCudaErrors(cudaMalloc(&d_scene, sizeof(hitable*)));
        camera** d_cam;
        checkCudaErrors(cudaMalloc(&d_cam, sizeof(camera*)));
        createScene<<<1, 1>>>(d_scene, d_cam, width, height, &d_rand_state[0]);
        checkCudaErrors(cudaDeviceSynchronize());
        size_t* d_thread_pixel_nums;
        cudaStream_t stream0;
        cudaStream_t stream1;
        int priority_low, priority_high;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
        checkCudaErrors(cudaStreamCreateWithPriority(&stream0, cudaStreamNonBlocking, priority_low));
        checkCudaErrors(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, priority_high));
        checkCudaErrors(cudaMalloc(&d_thread_pixel_nums, sizeof(size_t) * threads * blocks));
        renderKernel<<<blocks, threads, 0, stream0>>>(width, height, d_cam, d_pixels, d_scene, threads * blocks, ns, d_rand_state, d_thread_pixel_nums);
        long start = clock();
        printf("get\n");
        size_t* d_all_finished_pixels;
        checkCudaErrors(cudaMalloc(&d_all_finished_pixels, sizeof(size_t)));
        size_t finished_pixels = 0;
        size_t totalPixels = width * height;
        while (finished_pixels < totalPixels) {
            pixelCount<<<1, 1, 0, stream1>>>(d_thread_pixel_nums, threads * blocks, width * height, d_all_finished_pixels);
            printf("pixels\n");
            cudaMemcpyAsync(&finished_pixels, d_all_finished_pixels, sizeof(size_t), cudaMemcpyDeviceToHost, stream1);
            cudaStreamSynchronize(stream1);
            printf("cudaMemcpy\n");
            printf("%d %d\n", finished_pixels, totalPixels);
            Sleep(500);
        }
        checkCudaErrors(cudaDeviceSynchronize());
        long stop = clock();
        printf("total time: %f s\n", static_cast<float>((stop - start)) / CLOCKS_PER_SEC);
        printf("kernel done\n");
        cudaMemcpy(host_pixels, d_pixels, 4 * width * height, cudaMemcpyDeviceToHost);
    }
}
