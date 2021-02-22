#pragma once
#include "material.h"
#include "texture.h"
#include "hitable.h"
#include "mesh.h"
#include "sphere.h"
#include "cudaMesh.h"
#include "rz_types.h"

#include "triangle.h"

class AssetFactor {
public:
	AssetFactor() {}
	~AssetFactor() {}
	AssetNode<sphere> createSphere(Point3f cen, float r, AssetNode<material> matr_node) {
		AssetNode<sphere> node;
		node.asset_host = new sphere(cen, r, matr_node.asset_host);
		checkCudaErrors(cudaMalloc(&node.asset_device, sizeof(sphere)));
		sphere temp(cen, r, matr_node.asset_device);
		checkCudaErrors(cudaMemcpy(node.asset_device, &temp, sizeof(sphere), cudaMemcpyHostToDevice));
		AssetNode<hitable> tnode;
		tnode.asset_host = node.asset_host;
		tnode.asset_device = node.asset_device;
		hitableNode.push_back(tnode);
		return node;
	}
	AssetNode<CudaMesh> createMesh(Mesh* mesh) {
		CudaMesh* cudaMesh = new CudaMesh(mesh);
		AssetNode<CudaMesh> node;
		node.asset_host = cudaMesh;
		CudaMesh temp(*cudaMesh);
		temp.trianglelist_size = cudaMesh->trianglelist_size;
		checkCudaErrors(cudaMalloc(&temp.trianglelist, sizeof(Triangle) * temp.trianglelist_size));
		checkCudaErrors(cudaMemcpy(temp.trianglelist, cudaMesh->trianglelist, sizeof(Triangle) * temp.trianglelist_size, cudaMemcpyHostToDevice));
		temp.vertex_size = cudaMesh->vertex_size;
		checkCudaErrors(cudaMalloc(&temp.vertex, sizeof(Vertex) * temp.vertex_size));
		checkCudaErrors(cudaMemcpy(temp.vertex, cudaMesh->vertex, sizeof(Vertex) * temp.vertex_size, cudaMemcpyHostToDevice));
		temp.vt_size = cudaMesh->vt_size;
		checkCudaErrors(cudaMalloc(&temp.vt, sizeof(Point2f) * temp.vertex_size));
		checkCudaErrors(cudaMemcpy(temp.vt, cudaMesh->vt, sizeof(Point2f) * temp.vt_size, cudaMemcpyHostToDevice));
		temp.vn_size = cudaMesh->vn_size;
		checkCudaErrors(cudaMalloc(&temp.vn, sizeof(Vector3f) * temp.vn_size));
		checkCudaErrors(cudaMemcpy(temp.vn, cudaMesh->vn, sizeof(Vector3f) * temp.vn_size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc(&node.asset_device, sizeof(CudaMesh)));
		checkCudaErrors(cudaMemcpy(node.asset_device, &temp, sizeof(CudaMesh), cudaMemcpyHostToDevice));
		meshNode.push_back(node);
		AssetNode<hitable> tnode;
		tnode.asset_host = node.asset_host;
		tnode.asset_device = node.asset_device;
		hitableNode.push_back(tnode);
		return node;
	}
	AssetNode<solid_texture> createSolidTexture(const Color& col) {
		AssetNode<solid_texture> node;
		node.asset_host = new solid_texture(col);
		checkCudaErrors(cudaMalloc(&node.asset_device, sizeof(solid_texture)));
		checkCudaErrors(cudaMemcpy(node.asset_device, node.asset_host, sizeof(solid_texture), cudaMemcpyHostToDevice));
		registTexture(node.asset_host, node.asset_device);
		return node;
	}
	AssetNode<checker_texture> createCheckerTexture(AssetNode<rdxr_texture> texture0, AssetNode<rdxr_texture> texture1) {
		AssetNode<checker_texture> node;
		node.asset_host = new checker_texture(texture0.asset_host, texture1.asset_host);
		checkCudaErrors(cudaMalloc(&node.asset_device, sizeof(checker_texture)));
		checker_texture temp(texture0.asset_device, texture1.asset_device);
		checkCudaErrors(cudaMemcpy(node.asset_device, &temp, sizeof(checker_texture), cudaMemcpyHostToDevice));
		registTexture(node.asset_host, node.asset_device);
		return node;
	}
	AssetNode<image_texture> createImageTexture(const std::string& image_path) {
		AssetNode<image_texture> node;
		node.asset_host = new image_texture(image_path);
		checkCudaErrors(cudaMalloc(&node.asset_device, sizeof(image_texture)));
		checkCudaErrors(cudaMemcpy(node.asset_device, node.asset_host, sizeof(image_texture), cudaMemcpyHostToDevice));
		registTexture(node.asset_host, node.asset_device);
		return node;
	}
	AssetNode<lambertian> createLambertian(AssetNode<rdxr_texture> _texture, float light = 0, Vector3f ka = 0, Vector3f ks = 0, Vector3f ke = 0, Vector3f kd = 0, float ni = 0, float ns = 0, float diffuse = 0, float illum = 0) {
		AssetNode<lambertian> node;
		node.asset_host = new lambertian(_texture.asset_host);
		node.asset_host->rdx_light = light;
		node.asset_host->ka = ka;
		node.asset_host->ks = ks;
		node.asset_host->ke = ke;
		node.asset_host->kd = kd;
		node.asset_host->ni = ni;
		node.asset_host->ns = ns;
		node.asset_host->d = diffuse;
		node.asset_host->illum = illum;
		checkCudaErrors(cudaMalloc(&node.asset_device, sizeof(lambertian)));
		lambertian temp(*node.asset_host);
		temp.albedo = _texture.asset_device;
		checkCudaErrors(cudaMemcpy(node.asset_device, &temp, sizeof(lambertian), cudaMemcpyHostToDevice));
		registMaterial(node.asset_host, node.asset_device);
		return node;
	}
	AssetNode<diffuse_light> createDiffuseLight(AssetNode<rdxr_texture> _texture) {
		AssetNode<diffuse_light> node;
		node.asset_host = new diffuse_light(_texture.asset_host);
		checkCudaErrors(cudaMalloc(&node.asset_device, sizeof(diffuse_light)));
		diffuse_light temp(_texture.asset_device);
		checkCudaErrors(cudaMemcpy(node.asset_device, &temp, sizeof(lambertian), cudaMemcpyHostToDevice));
		registMaterial(node.asset_host, node.asset_device);
		return node;
	}
	void getTexture(AssetNode<rdxr_texture>** _texture, size_t* size) {
		*_texture = textureNode.data();
		*size = textureNode.size();
	}
	void getMaterial(AssetNode<material>** _material, size_t* size) {
		*_material = materialNode.data();
		*size = materialNode.size();
	}
	void getMesh(AssetNode<CudaMesh>** _mesh, size_t* size) {
		*_mesh = meshNode.data();
		*size = meshNode.size();
	}
	void getHitable(AssetNode<hitable>** _hitable, size_t* size) {
		*_hitable = hitableNode.data();
		*size = hitableNode.size();
	}

private:
	std::vector<AssetNode<rdxr_texture>> textureNode;
	std::vector<AssetNode<material>> materialNode;
	std::vector<AssetNode<CudaMesh>> meshNode;
	std::vector<AssetNode<hitable>> hitableNode;

	void registTexture(rdxr_texture* data_host, rdxr_texture* data_device) {
		AssetNode<rdxr_texture> node;
		node.asset_host = data_host;
		node.asset_device = data_device;
		textureNode.push_back(node);
	}
	void registMaterial(material* data_host, material* data_device) {
		AssetNode<material> node;
		node.asset_host = data_host;
		node.asset_device = data_device;
		materialNode.push_back(node);
	}
	void registMesh(CudaMesh* data_host, CudaMesh* data_device) {
		AssetNode<CudaMesh> node;
		node.asset_host = data_host;
		node.asset_device = data_device;
		meshNode.push_back(node);
	}
	void registHitable(hitable* data_host, hitable* data_device) {
		AssetNode<hitable> node;
		node.asset_host = data_host;
		node.asset_device = data_device;
		hitableNode.push_back(node);
	}
};

__global__ void test(hitable** hitables, size_t size);
void f();

