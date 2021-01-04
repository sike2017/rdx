#pragma once
#include <random>
#include <time.h>
#include "math/monolith_math.h"

float rdx_rand() {
	return rand() % 10001 / 10000.0;
}

void rdx_srand(unsigned int seed) {
	std::default_random_engine random_engine;
	std::uniform_real_distribution<float> uniform_dist(0, 1);
	random_engine.seed(seed);
}

Vector3f random_in_unit_sphere() {
	Vector3f p;
	rdx_srand(time(NULL));
	do {
		p = 2.0 * Vector3f(rdx_rand(), rdx_rand(), rdx_rand()) - Vector3f(1, 1, 1);
	} while (p.squared_length() >= 1.0);
	return p;
}
