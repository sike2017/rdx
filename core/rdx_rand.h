#ifndef __RDX_RAND_H__
#define __RDX_RAND_H__

#include <curand_kernel.h>
#include <random>

__host__ __device__ inline float rdx_rand(curandState* state) {
#if defined(__CUDA_ARCH__)
	return curand_uniform(state);
#else
	return rand() % 10001 / 10000.0;
#endif
}

__host__ __device__ inline void rdx_srand() {
#ifndef __CUDA_ARCH__
	srand(time(NULL));
#endif
}

#endif