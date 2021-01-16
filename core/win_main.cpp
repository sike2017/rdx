#include "rdx.h"

int WinMain(HINSTANCE hinstance, HINSTANCE prevHinstance, LPSTR cmd, int nCmd) {
	RDXWindow window(hinstance, prevHinstance, cmd, nCmd, 1024, 720, L"rdx raytracer");
	return window.display();
}
