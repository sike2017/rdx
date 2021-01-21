#ifndef __RDX_RAND_H__
#define __RDX_RAND_H__
#include <curand_kernel.h>

__device__ inline float rdx_rand() {
	curandState state;
	return curand_uniform(&state);
}

#endif