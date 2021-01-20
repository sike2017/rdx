#pragma once
#include <Windows.h>
#include <thread>
#include "Painter.h"
#include "log/log.h"

extern Painter painter;

// do not using this namespace in your program!
namespace display
{
	extern int screen_keys[512];
	LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
}

enum class RENDER_COMMAND
{
	RUNNING,
	STOP
};

class WindowDisplayer
{
public:
	WindowDisplayer(HINSTANCE thInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow, int twidth, int theight, const wchar_t* twindowName);
	~WindowDisplayer() {}

	int display();
	void setPixel(int x, int y, RGBA color);
	void update();
	
	int width() const;
	int height() const;

protected:
	int colorIndex;

	int _width, _height;

	enum class RENDER_STATUS
	{
		CALL_NEXTTIME,
		CALL_STOP,
		CALL_STOP_SAVE_IMAGE
	};

	RENDER_STATUS renderStatus;

	virtual RENDER_STATUS render(RENDER_COMMAND *renderCommand) {
		rlog.print("virtual render\n");

		RGBA color0(0, 162, 232, 0);
		RGBA color1(112, 146, 190, 0);

		int size = _width * _height;
		for (int i = 0; i < size; i++)
		{
			int x = i % _width;
			int y = i / _width;
			if (!colorIndex)
			{
				setPixel(x, y, color0);
			}
			else {
				setPixel(x, y, color1);
			}
			if (*renderCommand == RENDER_COMMAND::STOP) {
				return RENDER_STATUS::CALL_STOP;
			}
		}

		colorIndex = !colorIndex;

		update();
		return RENDER_STATUS::CALL_NEXTTIME;
	}
	virtual void keyboardEvent(int* screen_keys) {}
	RenderBitmap getRenderBitmap();

private:
	HINSTANCE hInstance;
	HINSTANCE hPrevInstance;
	LPSTR szCmdLine;
	int iCmdShow;
	std::wstring windowName;
	
	RenderBitmap rb;
	RENDER_COMMAND renderCommand;
	int InitWindow();
};
