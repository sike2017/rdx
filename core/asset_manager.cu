#include "asset_manager.h"
#include "parser/file_parser.h"

__global__ void test(hitable** hitables, size_t size) {
	Ray r;
	hit_record rec;
	printf("get\n");
	for (int index = 0; index < size; index++) {
		hitables[index]->hit(r, 0.01, FLT_MAX, &rec);
		printf("hello\n");
	}
}

void f() {

	Mesh mesh;
	AssetFactor assetFactor;
	ObjParser pr(&assetFactor);
	pr.parse("model/cube.obj", &mesh);

	assetFactor.createMesh(&mesh);
	AssetNode<rdxr_texture>* node;
	size_t size;
	assetFactor.getTexture(&node, &size);
	std::vector<hitable*> hitables;
	AssetNode<hitable>* hitableList;
	size_t list_size;
	assetFactor.getHitable(&hitableList, &list_size);
	for (int index = 0; index < list_size; index++) {
		hitables.push_back(hitableList[index].asset_device);
	}
	hitable** d_hitables;
	checkCudaErrors(cudaMalloc(&d_hitables, sizeof(hitable*) * list_size));
	checkCudaErrors(cudaMemcpy(d_hitables, hitables.data(), sizeof(hitable*) * list_size, cudaMemcpyHostToDevice));
	printf("get\n");
	test<<<10,10>>>(d_hitables, list_size);
	printf("done\n");
}
