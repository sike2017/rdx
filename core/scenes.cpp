#include "scenes.h"
#include "hitablelist.h"
#include "parser/file_parser.h"
#include "sphere.h"
#include "material.h"
#include "bvh.h"
#include <thrust/device_vector.h>

hitable* random_scene() {
	int n = 500;
	hitable** list = new hitable * [n + 1];
	rdxr_texture* checker = new checker_texture(new solid_texture(Color(0.2, 0.3, 0.1)), new solid_texture(Color(0.9, 0.9, 0.9)));
	list[0] = new sphere(Vector3f(0, -1000, 0), 1000, new lambertian(checker));
	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = rdx_rand();
			Vector3f center(a + 0.9 * rdx_rand(), 0.2, b + 0.9 * rdx_rand());
			if ((center - Vector3f(4.0, 2.0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					list[i++] = new sphere(center, 0.2, new lambertian(new solid_texture(Color(rdx_rand() * rdx_rand(), rdx_rand() * rdx_rand(), rdx_rand() * rdx_rand()))));
				}
				else if (choose_mat < 0.95) {
					list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
	}

	list[i++] = new sphere(Vector3f(0, 1, 0), 1.0, new dielectric(1.5));
	list[i++] = new sphere(Vector3f(-4, 1, 0), 1.0, new lambertian(new solid_texture(Color(0.4, 0.2, 0.1))));
	list[i++] = new sphere(Vector3f(4, 1, 0), 1.0, new metal(Vector3f(0.7, 0.6, 0.5), 0.0));

	hitable* scene = new hitable_list(list, i);
	return scene;
}

hitable* simple_light() {
	rdxr_texture* checker = new checker_texture(new solid_texture(Color(0.2, 0.3, 0.1)), new solid_texture(Color(0.9, 0.9, 0.9)));
	hitable** list = new hitable * [4];
	list[0] = new sphere(Vector3f(0, -1000, 0), 1000, new lambertian(checker));
	list[1] = new sphere(Vector3f(0, 2, 0), 2, new lambertian(checker));
	list[2] = new sphere(Vector3f(0, 7, 0), 2, new diffuse_light(new solid_texture(Color(4, 4, 4))));
	list[3] = new sphere(Vector3f(1, 4, 2), 2, new diffuse_light(new solid_texture(Color(4, 4, 4))));
	hitable* scene = new hitable_list(list, 4);
	return scene;
}

hitable* cornell_box() {
	Mesh mesh(nullptr);
	ObjParser objParser;
	objParser.parse("model/cornell_box.obj", &mesh);
	sphere* model = new sphere(Vector3f(0, 0, 0), 2, new diffuse_light(new solid_texture(Color(4, 4, 4))));
	hitable** list = new hitable * [2];
	list[0] = &mesh;
	list[1] = model;
	hitable* scene = new hitable_list(list, 2);
	return scene;
}

hitable* cube_light() {
	Mesh mesh(nullptr);
	ObjParser objParser;
	objParser.parse("model/cube_light.obj", &mesh);
	hitable** list = new hitable * [4];
	list[0] = &mesh;
	list[1] = new sphere(Vector3f(2, 5, 0), 1.0, new dielectric(1.7));
	list[2] = new sphere(Vector3f(0, -1000, 0), 1000, new lambertian(new checker_texture(new solid_texture(Color(0.12, 0.19, 0.25)), new solid_texture(Color(0.9, 0.9, 0.9)))));
	list[3] = new sphere(Vector3f(0, 20, 0), 2, new diffuse_light(new solid_texture(Color(40, 40, 40))));
	hitable* scene = new hitable_list(list, 4);
	return scene;
}

__global__ void spot(hitable_list* list) {
	hitable** pphitable = new hitable * [7];
	Mesh* mesh = new Mesh(new lambertian(new solid_texture(Color(0.4, 0.2, 0.1))));
	ObjParser objp;
	objp.parse("model/spot/spot_triangulated_good.obj", mesh);
	pphitable[0] = mesh;
	pphitable[1] = new sphere(Vector3f(0, 20, 0), 2, new diffuse_light(new solid_texture(Color(40, 40, 40))));
	pphitable[2] = new sphere(Vector3f(4.21, 0, 0), 1, new dielectric(1.24));
	pphitable[3] = new sphere(Vector3f(0, -1001, 0), 1000, new lambertian(new checker_texture(new solid_texture(Color(0.12, 0.19, 0.25)), new solid_texture(Color(0.9, 0.9, 0.9)))));
	Mesh* cube = new Mesh(new lambertian(new solid_texture(Color(0.7, 0.7, 0.7))));
	objp.parse("model/cube.obj", cube);
	cube->trans(2, 0, 0);
	pphitable[4] = cube;
	pphitable[5] = new sphere(Vector3f(-2.0, 0, 0), 1, new metal(Vector3f(0.9, 0.9, 0.9), 0.1));
	pphitable[6] = new sphere(Vector3f(-4.16, 0, 0), 1, new lambertian(new solid_texture(Color(0.4, 0.2, 0.1))));
	*list = hitable_list(pphitable, 7);
}