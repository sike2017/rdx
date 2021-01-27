#ifndef __RDX_RAND_H__
#define __RDX_RAND_H__
#include <curand_kernel.h>
#include <random>

inline float host_rand() {
	return rand() % 10001 / 10000.0;
}

inline void host_srand(unsigned int seed) {
	srand(seed);
}

__device__ inline float device_rand(curandState* state) {
	return curand_uniform(state);
}


#endif