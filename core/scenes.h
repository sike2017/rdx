#pragma once
#include "rdx_random.h"
#include "sphere.h"
#include "hitable.h"
#include "material.h"
#include "hitablelist.h"

hitable* random_scene() {
	int n = 500;
	hitable** list = new hitable * [n + 1];
	list[0] = new sphere(Vector3f(0, -1000, 0), 1000, new lambertian(Vector3f(0.5, 0.5, 0.5)));
	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = rdx_rand();
			Vector3f center(a + 0.9 * rdx_rand(), 0.2, b + 0.9 * rdx_rand());
			if ((center - Vector3f(4.0, 2.0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					list[i++] = new sphere(center, 0.2, new lambertian(Vector3f(rdx_rand() * rdx_rand(), rdx_rand() * rdx_rand(), rdx_rand() * rdx_rand())));
				}
				else if (choose_mat < 0.95) {
					list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
	}

	list[i++] = new sphere(Vector3f(0, 1, 0), 1.0, new dielectric(1.5));
	list[i++] = new sphere(Vector3f(-4, 1, 0), 1.0, new lambertian(Vector3f(0.4, 0.2, 0.1)));
	list[i++] = new sphere(Vector3f(4, 1, 0), 1.0, new metal(Vector3f(0.7, 0.6, 0.5), 0.0));

	return new hitable_list(list, i);
}
