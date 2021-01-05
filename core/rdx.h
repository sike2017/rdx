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

class RDXWindow : public WindowDisplayer {
public:
	RDXWindow(HINSTANCE thInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow, int _width, int _height, const wchar_t* twindowName)
		: WindowDisplayer(thInstance, hPrevInstance, szCmdLine, iCmdShow, _width, _height, twindowName) {
		ns = 100;
	}
	~RDXWindow() {}

protected:
	virtual RENDER_STATUS render(RENDER_COMMAND* renderCommand) override {
		hitable* world = random_scene();
		Vector3f lookfrom(13, 2, 3);
		Vector3f lookat(0, 0, 0);
		//hitable* list[4];
		//list[0] = new sphere(Vector3f(0, 0, -1), 0.5, new lambertian(Vector3f(0.1, 0.2, 0.5)));
		//list[1] = new sphere(Vector3f(0, -100.5, -1), 100, new lambertian(Vector3f(0.8, 0.8, 0.0)));
		//list[2] = new sphere(Vector3f(1, 0, -1), 0.5, new metal(Vector3f(0.8, 0.6, 0.2), 0.3));
		//list[3] = new sphere(Vector3f(-1, 0, -1), 0.5, new dielectric(1.5));
		//list[3] = new sphere(Vector3f(-1, 0, -1), -0.45, new dielectric(1.5));
		//hitable* world = new hitable_list(list, 4);
		camera cam(lookfrom, lookat, Vector3f(0, 1, 0), 20, static_cast<float>(width()) / static_cast<float>(height()));
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
