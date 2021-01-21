#include "rdx.h"
#include "rdx_cuda/rdx_cuda.h"

int main(int argc, char* argv) {
	unsigned long long width = 1920;
	unsigned long long height = 1080;
	RdxCuda window(GetModuleHandle(NULL), NULL, argv, argc, width, height, L"rdx raytracer");
	return window.display();
}
