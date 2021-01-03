#include "Painter.h"

RGBA::RGBA()
{
	red = 0;
	green = 0;
	blue = 0;
	alpha = 0;
}

RGBA::RGBA(uint8_t tred, uint8_t tgreen, uint8_t tblue, uint8_t talpha)
{
	red = tred;
	green = tgreen;
	blue = tblue;
	alpha = talpha;
}

uint32_t RGBA::toRGBAUint32()
{
	uint32_t result;
	uint8_t* p = reinterpret_cast<uint8_t*>(&result);

	p[0] = red;
	p[1] = green;
	p[2] = blue;
	p[3] = alpha;

	return result;
}

uint32_t RGBA::toBGRAUint32()
{
	uint32_t result;
	uint8_t* p = reinterpret_cast<uint8_t*>(&result);

	p[0] = blue;
	p[1] = green;
	p[2] = red;
	p[3] = alpha;

	return result;
}

RGBA RGBA::operator+(const RGBA& color) const
{
	RGBA result = *this;
	unsigned int v;
	v = result.red + color.red;
	if (v > 255)
	{
		result.red = 255;
	}
	else
	{
		result.red = v;
	}
	v = result.green + color.green;
	if (v > 255)
	{
		result.green = 255;
	}
	else
	{
		result.green = v;
	}
	v = result.blue + color.blue;
	if (v > 255)
	{
		result.blue = 255;
	}
	else
	{
		result.blue = v;
	}
	v = result.alpha + color.alpha;
	if (v > 255)
	{
		result.alpha = 255;
	}
	else
	{
		result.alpha = v;
	}
	return result;
}

RGBA RGBA::operator-(const RGBA& color) const
{
	RGBA result = *this;
	unsigned int v;
	v = result.red - color.red;
	if (v < 0)
	{
		result.red = 0;
	}
	else
	{
		result.red = v;
	}
	v = result.green - color.green;
	if (result.green < 0)
	{
		result.green = 0;
	}
	else
	{
		result.green = v;
	}
	v = result.blue - color.blue;
	if (v < 0)
	{
		result.blue = 0;
	}
	else
	{
		result.blue = v;
	}
	v = result.alpha - color.alpha;
	if (v < 0)
	{
		result.alpha = 0;
	}
	else
	{
		result.alpha = v;
	}
	return result;
}

RGBA RGBA::operator=(const RGBA& color)
{
	this->red = color.red;
	this->green = color.green;
	this->blue = color.blue;
	this->alpha = color.alpha;

	return *this;
}

bool RGBA::operator==(const RGBA& color) const
{
	return (color.red == this->red && color.green == this->green && color.blue == this->blue && color.alpha == this->alpha);
}

RGBA RGBA::mul(float u) const
{
	float v;
	RGBA color;
	v = u * red;
	if (v > 255.0f)
	{
		color.red = 255;
	}
	else if (v < 0.0f)
	{
		color.red = 0;
	}
	else
	{
		color.red = v;
	}
	v = u * green;
	if (v > 255.0f)
	{
		color.green = 255;
	}
	else if (v < 0.0f)
	{
		color.green = 0;
	}
	else
	{
		color.green = v;
	}
	v = u * blue;
	if (v > 255.0f)
	{
		color.blue = 255;
	}
	else if (v < 0.0f)
	{
		color.blue = 0;
	}
	else
	{
		color.blue = v;
	}
	v = u * alpha;
	if (v > 255.0f)
	{
		color.alpha = 255;
	}
	else if (v < 0.0f)
	{
		color.alpha = 0;
	}
	else
	{
		color.alpha = v;
	}
	return color;
}

RGBA operator*(const float& left, const RGBA& right) {
	return right.mul(left);
}

unique_ptr<wchar_t[]> ConvertStringToLPCWSTR(string& str)
{
	size_t length = str.length();
	int size = MultiByteToWideChar(CP_ACP, 0, str.c_str(), length, NULL, 0);
	wchar_t* buffer = new wchar_t[size + 1];
	MultiByteToWideChar(CP_UTF8, 0, str.c_str(), length, buffer, size);
	buffer[size] = '\0';
	return unique_ptr<wchar_t[]>(buffer);
}

RenderBitmap::RenderBitmap(int width, int height)
{
	assert(width > 0 && height > 0);
	bmColorSize = 4;
	bmPlanes = 1;
	bmWidth = width;
	bmHeight = height;
	hBitmap = NULL;
	pBmBits = NULL;

	BITMAPINFOHEADER bmphdr = { 0 };
	bmphdr.biSize = sizeof(bmphdr);
	bmphdr.biWidth = width;
	bmphdr.biHeight = -height;
	bmphdr.biPlanes = 1;
	bmphdr.biBitCount = 32;
	bmphdr.biCompression = BI_RGB;
	bmphdr.biSizeImage = width * height * 4;

	hBitmap = CreateDIBSection(NULL, reinterpret_cast<PBITMAPINFO>(&bmphdr), DIB_RGB_COLORS,
		reinterpret_cast<void**>(&pBmBits), NULL, 0);
	if (!hBitmap)
	{
		rlog.print("CreateDIBSection failed\n");
		return;
	}
}

RenderBitmap::~RenderBitmap()
{
	cleanUp();
}

bool RenderBitmap::loadBitmapFromFile(string fileName)
{
	unique_ptr<wchar_t[]> p = ConvertStringToLPCWSTR(fileName);
	HANDLE loadResult = LoadImage(NULL, p.get(), IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
	if (!loadResult)
	{
		rlog.print("load failed\n");
		return false;
	}
	hBitmap = static_cast<HBITMAP>(loadResult);
	BITMAP bitmap;

	int result = GetObject(hBitmap, sizeof(BITMAP), &bitmap);

	bmPlanes = bitmap.bmPlanes;
	bmColorSize = bitmap.bmBitsPixel;
	bmWidth = bitmap.bmWidth;
	bmHeight = bitmap.bmHeight;

	return true;
}

HBITMAP RenderBitmap::getHBitmap()
{
	return hBitmap;
}

int RenderBitmap::getWidth()
{
	return bmWidth;
}

int RenderBitmap::getHeight()
{
	return bmHeight;
}

void RenderBitmap::setPixel(int x, int y, RGBA color)
{
	assert(x >= 0 && y >= 0 && x < bmWidth && y < bmHeight);

	x = convertSoftwareCoordinateToDeviceCoordinateX(x);
	y = convertSoftwareCoordinateToDeviceCoordinateY(y);
	pBmBits[y * bmWidth + x] = color.toBGRAUint32();
}

bool RenderBitmap::cleanUp()
{
	return DeleteObject(hBitmap);
}

bool RenderBitmap::saveBitmap(const char* fileName)
{
	image_png::image_t image = image_png::create_image(bmWidth, bmHeight);
	uint32_t* pBGRA = pBmBits;
	uint32_t* pRGBA = reinterpret_cast<uint32_t *>(image.p_buffer);
	int p;
	for (p = 0; p < bmWidth * bmHeight; p++) {
		*pRGBA = bgra_to_rgba(*pBGRA);
		pRGBA++;
		pBGRA++;
	}
	image_png::write_image(fileName, &image);

	return true;
}

int RenderBitmap::convertSoftwareCoordinateToDeviceCoordinateX(int x)
{
	return x;
}

int RenderBitmap::convertSoftwareCoordinateToDeviceCoordinateY(int y)
{
	return (bmHeight - y - 1);
}

uint32_t RenderBitmap::bgra_to_rgba(uint32_t& bgra)
{
	uint32_t rgba = bgra;

	reinterpret_cast<uint8_t*>(&rgba)[0] = reinterpret_cast<uint8_t*>(&bgra)[2];
	reinterpret_cast<uint8_t*>(&rgba)[2] = reinterpret_cast<uint8_t*>(&bgra)[0];
	return rgba;
}

Painter::Painter()
{
	m_hwnd = NULL;
	ZeroMemory(&m_PaintStruct, sizeof(m_PaintStruct));
	m_hInstance = NULL;
	m_hdc = NULL;
	ZeroMemory(&m_bitmap, sizeof(m_bitmap));
	m_pFrameBitmap = nullptr;
}

Painter::~Painter()
{
	rlog.print("Painter destructor called\n");
}

bool Painter::Initialize(HWND hwnd)
{

	m_hwnd = hwnd;
	LONG_PTR result = GetWindowLongPtr(hwnd, GWLP_HINSTANCE);
	if (!result)
	{
		rlog.print("initialize failed\n");
		DWORD e = GetLastError();
		rlog.print("last error: %ld\n", e);
		return false;
	}
	m_hInstance = reinterpret_cast<HINSTANCE>(result);
	return true;
}

void Painter::SetFrameBitmapBuffer(RenderBitmap* bitmap)
{
	m_pFrameBitmap = bitmap;
}

bool Painter::PainterBeginPaint()
{
	m_hdc = BeginPaint(m_hwnd, &m_PaintStruct);
	if (m_hdc) {
		return true;
	}
	else {
		return false;
	}
}

bool Painter::PainterEndPaint()
{
	return static_cast<bool>(EndPaint(m_hwnd, &m_PaintStruct));
}

bool Painter::DrawBitmapFromFile(string fileName)
{
	HBITMAP hBitmap;

	unique_ptr<wchar_t[]> p = ConvertStringToLPCWSTR(fileName);

	hBitmap = static_cast<HBITMAP>(LoadImage(NULL, p.get(), IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE));

	GetObject(hBitmap, sizeof(BITMAP), &m_bitmap);

	int cxSource = m_bitmap.bmWidth;
	int cySource = m_bitmap.bmHeight;

	HDC hdcMem = PainterCreateCompatibleDC();
	if (!hdcMem)
	{
		return false;
	}

	SelectObject(hdcMem, hBitmap);
	BitBlt(m_hdc, 5, 20, cxSource, cySource, hdcMem, 0, 0, SRCCOPY);

	if (!PainterDeleteDC(hdcMem))
	{
		return false;
	}

	return true;

}

bool Painter::UpdateFrame()
{
	InvalidateRect(m_hwnd, nullptr, true);
	UpdateWindow(m_hwnd);
	return true;
}

bool Painter::DrawFrame()
{	
	HDC hdcMem = PainterCreateCompatibleDC();
	if (!hdcMem)
	{
		return false;
	}

	HBITMAP hBitmap = m_pFrameBitmap->getHBitmap();
	SelectObject(hdcMem, hBitmap);
	bool result = BitBlt(m_hdc, 0, 0, m_pFrameBitmap->getWidth(), m_pFrameBitmap->getHeight(), hdcMem, 0, 0, SRCCOPY);

	if (!result)
	{
		return false;
	}

	if (!PainterDeleteDC(hdcMem))
	{
		return false;
	}

	return true;
}

HWND Painter::GetHWND()
{
	return m_hwnd;
}

HDC Painter::PainterCreateCompatibleDC()
{
	return CreateCompatibleDC(m_hdc);
}

bool Painter::PainterDeleteDC(HDC hdc)
{
	return static_cast<bool>(DeleteDC(hdc));
}
