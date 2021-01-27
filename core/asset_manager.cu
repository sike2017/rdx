#include "asset_manager.h"
#include "parser/file_parser.h"

void f() {
	Mesh mesh;
	AssetFactor assetFactor;
	ObjParser pr(&assetFactor);
	pr.parse("model/cube.obj", &mesh);

	assetFactor.createMesh(&mesh);
	AssetNode<rdxr_texture>* node;
	size_t size;
	assetFactor.getTexture(&node, &size);
	printf("%p\n", node->asset_device);
}
