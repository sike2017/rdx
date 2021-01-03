#pragma once
#include <Windows.h>
#include <assert.h>
#include "log/log.h"
#include "image_png.h"

using std::string;
using std::unique_ptr;

class RGBA
{
public:
	RGBA();
	RGBA(uint8_t tred, uint8_t tgreen, uint8_t tblue, uint8_t talpha = 255);

	uint32_t toRGBAUint32();
	uint32_t toBGRAUint32();

	RGBA operator+(const RGBA& color) const;
	RGBA operator-(const RGBA& color) const;
	RGBA operator=(const RGBA& color);
	bool operator==(const RGBA& color) const;
	RGBA mul(float u) const;

	uint8_t red;
	uint8_t green;
	uint8_t blue;
	uint8_t alpha;
};

RGBA operator*(const float& left, const RGBA& right);

class RenderBitmap
{
public:
	RenderBitmap() = delete;
	RenderBitmap(int width, int height);
	~RenderBitmap();

	bool loadBitmapFromFile(string fileName);
	HBITMAP getHBitmap();
	int getWidth();
	int getHeight();
	void setPixel(int x, int y, RGBA color);
	bool cleanUp();
	bool saveBitmap(const char* fileName);

private:
	int convertSoftwareCoordinateToDeviceCoordinateX(int x);
	int convertSoftwareCoordinateToDeviceCoordinateY(int y);
	uint32_t bgra_to_rgba(uint32_t& rgba);
	int bmPlanes;           // The number of bitmap plans.
	int bmColorSize;        // The number of bits required to indicate the color of a pixel.
	int bmWidth;            // The width, in pixels, of the bitmap. The width must be greater than zero.
	int bmHeight;           // The height, in pixels, of the bitmap. The height must be greater than zero.
	HBITMAP hBitmap;
	uint32_t* pBmBits;
};

class Painter
{
public:
	Painter();
	~Painter();

	bool Initialize(HWND hwnd);
	// 设置读取图像的地址
	void SetFrameBitmapBuffer(RenderBitmap* bitmap);
	bool PainterBeginPaint();
	bool PainterEndPaint();
	bool DrawBitmapFromFile(string fileName = "");

	bool UpdateFrame();
	bool DrawFrame();
	HWND GetHWND();

	HWND m_hwnd;
	HINSTANCE m_hInstance;
	PAINTSTRUCT m_PaintStruct;
	HDC m_hdc;
	BITMAP m_bitmap;
	RenderBitmap *m_pFrameBitmap;

	HDC PainterCreateCompatibleDC();
	bool PainterDeleteDC(HDC hdc);
};