#pragma once
#ifndef IMAGE_PNG_H
#define IMAGE_PNG_H

#include <memory.h>

namespace image_png {
	typedef struct image_struct {
		unsigned char* p_buffer;
		int width;
		int height;
		int channels_number;
	} image_t;

	const int CHANNELS_NUMBER = 4;

	image_t load_image(const char* filename);

	// create image, your must use free_image function to free image when you are done with the image.
	image_t create_image(int width, int height);

	// create image but do not create buffer, you do not need to free image when you are done with the image.
	image_t create_image_no_create_buffer(int width, int height, unsigned char* p_buffer);

	void free_image(image_t* image);

	void set_pixel(image_t* image, int x, int y, unsigned char r, unsigned char g, unsigned char b, unsigned char a);

	void write_image(const char* filename, image_t* image);
}

#endif
