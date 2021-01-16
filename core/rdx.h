#pragma once
#include <thread>
#include "rdx_random.h"
#include "display/WindowDisplayer.h"
#include "color.h"
#include "ray.h"
#include "hitable.h"
#include "material.h"
#include "hitablelist.h"
#include "camera.h"
#include "sphere.h"
#include "scenes.h"
#include "triangle.h"

class RDXWindow : public WindowDisplayer {
public:
	RDXWindow(HINSTANCE thInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow, int _width, int _height, const wchar_t* twindowName)
		: WindowDisplayer(thInstance, hPrevInstance, szCmdLine, iCmdShow, _width, _height, twindowName) {
		ns = 100;
	}
	~RDXWindow() {}

protected:
	virtual RENDER_STATUS render(RENDER_COMMAND* renderCommand) override {
		hitable* world = spot();
		//Vector3f lookfrom(278, 273, -800);
		//Vector3f lookat(278, 273, 0);
		Vector3f lookfrom(-1.0, 0, -12.4);
		Vector3f lookat(0, 0, 0);
		float dist_to_focus = (lookfrom - lookat).length();
		float aperture = 0.1;
		float fov = 40.0;
		camera cam(lookfrom, lookat, Vector3f(0, 1, 0), fov, static_cast<float>(width()) / static_cast<float>(height()), aperture, dist_to_focus);
		std::vector<std::thread> workers;
		int cpuNums = 6;
		std::vector<int> pixel_nums(cpuNums, 0);
		float rate = 0.f;
		unsigned long long screen_size = width() * height();
		for (int id = 0; id < cpuNums; id++) {
			workers.push_back(std::thread([=, &pixel_nums, &rate]() {
				int x, y;
				for (int index = id; index < screen_size; index += cpuNums) {
					if (*renderCommand == RENDER_COMMAND::STOP) {
						return;
					}
					x = index % width();
					y = index / width();
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
					pixel_nums[id]++;
				}
			}));
		}
		while (true) {
			if (*renderCommand == RENDER_COMMAND::STOP) {
				for (auto& worker : workers) {
					worker.join();
				}
				return RENDER_STATUS::CALL_STOP;
			}
			unsigned long long sum = 0;
			for (unsigned long long pixel_num : pixel_nums) {
				sum += pixel_num;
			}
			rate = (static_cast<float>(sum) / static_cast<float>(screen_size)) * 100;
			if (rate >= 100.f) {			
				rlog.print("%%100.00\n");
				for (auto& worker : workers) {
					worker.join();
				}
				break;
			}
			rlog.print("%%%.2f\n", rate);
			Sleep(500);
		}

		rlog << "done\n";

		update();
		return RENDER_STATUS::CALL_STOP_SAVE_IMAGE;
	}

private:
	int ns;

	Color color(const Ray& r, hitable* world, int depth) {
		hit_record rec;

		if (world->hit(r, 0.0f, FLT_MAX, &rec)) {
			Ray scattered;
			Vector3f attenuation;
			Color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
			if (depth < 50 && rec.mat_ptr->scatter(r, rec, &attenuation, &scattered)) {
				return emitted + attenuation * color(scattered, world, depth + 1);
			}
			else {
				return emitted;
			}
		}
		else {
			Vector3f unit_direction = unit_vector(r.direction());
			float t = 0.5 * (unit_direction.y() + 1.0);
			return (1.0 - t) * Vector3f(1.0, 1.0, 1.0) + t * Vector3f(0.5, 0.7, 1.0);
		}
	}

	RGBA to_rgba(const Color& color) {
		uint8_t ur = static_cast<int>(255.99 * range_float(color.r(), 0.0, 1.0));
		uint8_t ug = static_cast<int>(255.99 * range_float(color.g(), 0.0, 1.0));
		uint8_t ub = static_cast<int>(255.99 * range_float(color.b(), 0.0, 1.0));
		return RGBA(ur, ug, ub, 255);
	}
};

