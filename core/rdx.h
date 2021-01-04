#pragma once
#include "rdx_random.h"
#include "display/WindowDisplayer.h"
#include "color.h"
#include "ray.h"
#include "hitable.h"
#include "material.h"
#include "hitablelist.h"
#include "camera.h"
#include "sphere.h"

class RDXWindow : public WindowDisplayer {
public:
	RDXWindow(HINSTANCE thInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow, int _width, int _height, const wchar_t* twindowName)
		: WindowDisplayer(thInstance, hPrevInstance, szCmdLine, iCmdShow, _width, _height, twindowName) {
		ns = 100;
	}
	~RDXWindow() {}

protected:
	virtual RENDER_STATUS render() override {
		rdx_srand(time(NULL));
		Vector3f lower_left_corner(-2.0, -1.0, -1.0);
		Vector3f horizontal(4.0, 0.0, 0.0);
		Vector3f vertical(0.0, 2.0, 0.0);
		Vector3f origin(0.0, 0.0, 0.0);
		hitable* list[4];
		list[0] = new sphere(Vector3f(0, 0, -1), 0.5, new lambertian(Vector3f(0.1, 0.2, 0.5)));
		list[1] = new sphere(Vector3f(0, -100.5, -1), 100, new lambertian(Vector3f(0.8, 0.8, 0.0)));
		list[2] = new sphere(Vector3f(1, 0, -1), 0.5, new metal(Vector3f(0.8, 0.6, 0.2), 0.3));
		list[3] = new sphere(Vector3f(-1, 0, -1), 0.5, new dielectric(1.5));
		hitable* world = new hitable_list(list, 4);
		camera cam;
		for (int x = 0; x < width(); x++) {
			for (int y = 0; y < height(); y++) {
				Color col(0, 0, 0);
				for (int s = 0; s < ns; s++) {
					float u = static_cast<float>(x + rdx_rand()) / static_cast<float>(width());
					float v = static_cast<float>(y + rdx_rand()) / static_cast<float>(height());
					//setPixel(x, y, RGBA(0, 162, 232));
					Ray r = cam.get_ray(u, v);
					col += color(r, world, 0);
				}
				col /= static_cast<float>(ns);
				col = Color(sqrt(col.r()), sqrt(col.g()), sqrt(col.b()));
				setPixel(x, y, to_rgba(col));
			}
		}

		rlog << "done\n";

		update();
		return RENDER_STATUS::CALL_STOP_SAVE_IMAGE;
	}

private:
	int ns;

	Color color(const Ray& r, hitable* world, int depth) {
		hit_record rec;
		if (world->hit(r, 0.001, FLT_MAX, &rec)) {
			Ray scattered;
			Vector3f attenuation;
			if (depth < 50 && rec.mat_ptr->scatter(r, rec, &attenuation, &scattered)) {
				return attenuation * color(scattered, world, depth + 1);
			}
			else {
				return Color(0, 0, 0);
			}
		}
		else {
			Vector3f unit_direction = unit_vector(r.direction());
			float t = 0.5 * (unit_direction.y() + 1.0);
			return (1.0 - t) * Vector3f(1.0, 1.0, 1.0) + t * Vector3f(0.5, 0.7, 1.0);
		}
	}

	RGBA to_rgba(const Color& color) {
		uint8_t ur = static_cast<int>(255.99 * color.r());
		uint8_t ug = static_cast<int>(255.99 * color.g());
		uint8_t ub = static_cast<int>(255.99 * color.b());
		return RGBA(ur, ug, ub, 255);
	}
};
