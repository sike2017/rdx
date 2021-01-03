#pragma once
#include "display/WindowDisplayer.h"
#include "color.h"
#include "ray.h"

class RDXWindow : public WindowDisplayer {
public:
	RDXWindow(HINSTANCE thInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow, int _width, int _height, const wchar_t* twindowName)
		: WindowDisplayer(thInstance, hPrevInstance, szCmdLine, iCmdShow, _width, _height, twindowName) {}
	~RDXWindow() {}

protected:
	virtual RENDER_STATUS render() override {
		Vector3f lower_left_corner(-2.0, -1.0, -1.0);

		for (int x = 0; x < width(); x++) {
			for (int y = 0; y < height(); y++) {
				setPixel(x, y, RGBA(0, 162, 232));
			}
		}

		update();
		return RENDER_STATUS::CALL_STOP_SAVE_IMAGE;
	}

private:
	Color color(const Ray& r) {
		Vector3f unit_direction = unit_vector(r.direction());
		float t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * Vector3f(1.0, 1.0, 1.0) + t * Vector3f(0.5, 0.7, 1.0);
	}
};
