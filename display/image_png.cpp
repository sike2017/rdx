#include "image_png.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace image_png {
	image_t create_image(int width, int height)
	{
		size_t size = static_cast<long long>(width) * static_cast<long long>(height) * CHANNELS_NUMBER;
		unsigned char* pImage = new unsigned char[size];
		memset(pImage, 0, size);
		image_t image;
		image.p_buffer = pImage;
		image.width = width;
		image.height = height;
		image.channels_number = CHANNELS_NUMBER;

		return image;
	}

	image_t create_image_no_create_buffer(int width, int height, unsigned char* p_buffer)
	{
		size_t size = static_cast<long long>(width) * static_cast<long long>(height) * CHANNELS_NUMBER;
		image_t image;
		image.p_buffer = p_buffer;
		image.width = width;
		image.height = height;
		image.channels_number = CHANNELS_NUMBER;

		return image;
	}

	void free_image(image_t* image)
	{
		free(image->p_buffer);
	}

	void set_pixel(image_t* image, int x, int y, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
	{
		int index = (y * (image->width) + x) * image->channels_number;
		image->p_buffer[index] = r; // red
		image->p_buffer[index + 1] = g; // green
		image->p_buffer[index + 2] = b; // blue
		image->p_buffer[index + 3] = a; // alpha
	}

	void write_image(const char* filename, image_t* image)
	{
		stbi_write_png(filename, image->width, image->height, image->channels_number, image->p_buffer, image->width * image->channels_number);
	}
}
