#include "rdx_cuda/rdx_cuda.h"
#include "rdx.h"

int main(int argc, char* argv) {
	unsigned long long width = 200;
	unsigned long long height = 100;
	RdxCuda window(GetModuleHandle(NULL), NULL, argv, argc, width, height, L"rdx raytracer");

	return window.display();
}
