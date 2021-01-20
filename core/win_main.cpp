#include "rdx.h"
#include "rdx_cuda/render.cu"

int main(int argc, char* argv) {
	unsigned long long width = 1920;
	unsigned long long height = 1080;
	RDXWindow window(GetModuleHandle(NULL), NULL, argv, argc, width, height, L"rdx raytracer");
	//RdxCuda window1(GetModuleHandle(NULL), NULL, argv, argc, width, height, L"rdx raytracer");
	ra::RenderCuda render;
	RGBA* pixels = new RGBA[width * height];
	render.render(width, height, pixels, spot());
	return window.display();
}
