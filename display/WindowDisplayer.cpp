#include "WindowDisplayer.h"

Painter painter;
int display::screen_keys[512];

WindowDisplayer::WindowDisplayer(HINSTANCE thInstance, HINSTANCE thPrevInstance, LPSTR tszCmdLine, int tiCmdShow, int twidth, int theight, const wchar_t* twindowName) :
	hInstance(thInstance), hPrevInstance(thPrevInstance), szCmdLine(tszCmdLine), iCmdShow(tiCmdShow), _width(twidth), _height(theight), windowName(twindowName), rb(twidth, theight), 
	colorIndex(0), renderStatus(RENDER_STATUS::CALL_NEXTTIME)
{
	painter.SetFrameBitmapBuffer(&rb);
}

int WindowDisplayer::display()
{
	return InitWindow();
}

void WindowDisplayer::setPixel(int x, int y, RGBA color)
{
	rb.setPixel(x, y, color);
}

void WindowDisplayer::update()
{
	painter.UpdateFrame();
}

int WindowDisplayer::width() const {
	return _width;
}

int WindowDisplayer::height() const {
	return _height;
}

int WindowDisplayer::InitWindow()
{
	/*static TCHAR szAppName[] = TEXT("RayTracing");*/
	const TCHAR* szAppName = windowName.c_str();
	const TCHAR* szWindowName = windowName.c_str();
	HWND         hwnd;
	MSG          msg;
	WNDCLASS     wndclass = { 0 };

	ZeroMemory(&msg, sizeof(msg));

	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.lpfnWndProc = display::WndProc;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wndclass.lpszMenuName = NULL;
	wndclass.lpszClassName = szAppName;

	if (!RegisterClass(&wndclass))
	{
		MessageBox(NULL, TEXT("This program requires Windows NT!"),
			szAppName, MB_ICONERROR);
		return 0;
	}

	RECT rc = { 0, 0, _width, _height };
	AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, false);

	/*hwnd = CreateWindow(szAppName, TEXT("RayTracing"),*/
	hwnd = CreateWindow(szAppName, szWindowName,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT,
		rc.right - rc.left, rc.bottom - rc.top,
		NULL, NULL, hInstance, NULL);

	ShowWindow(hwnd, iCmdShow);
	UpdateWindow(hwnd);

	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{

			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else {
			switch (renderStatus) {
			case RENDER_STATUS::CALL_NEXTTIME:
				renderStatus = render();
				break;
			case RENDER_STATUS::CALL_STOP:
				break;
			case RENDER_STATUS::CALL_STOP_SAVE_IMAGE:
				rb.saveBitmap("image.png");
				renderStatus = RENDER_STATUS::CALL_STOP;
				break;
			}
			keyboardEvent(display::screen_keys);
		}
	}
	return msg.wParam;
}

LRESULT CALLBACK display::WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_CREATE:

		painter.Initialize(hwnd);
		return 0;

	case WM_KEYDOWN:
		screen_keys[wParam & 511] = 1;
		break;

	case WM_KEYUP:
		screen_keys[wParam & 511] = 0;
		break;

	case WM_PAINT:

		painter.PainterBeginPaint();
		painter.DrawFrame();
		painter.PainterEndPaint();

		return 0;

	case WM_ERASEBKGND:
		// ≤ª÷ÿªÊ±≥æ∞
		return 0;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hwnd, message, wParam, lParam);
}
