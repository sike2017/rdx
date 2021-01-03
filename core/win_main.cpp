#include "rdx.h"

int WinMain(HINSTANCE hinstance, HINSTANCE prevHinstance, LPSTR cmd, int nCmd) {
	RDXWindow window(hinstance, prevHinstance, cmd, nCmd, 200, 100, L"rdx raytracer");
	return window.display();
}
