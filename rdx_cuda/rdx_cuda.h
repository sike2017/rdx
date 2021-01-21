#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "display/WindowDisplayer.h"
#include "render.h"

class RdxCuda : public WindowDisplayer {
public:
    RdxCuda(HINSTANCE thInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow, int twidth, int theight, const wchar_t* twindowName) :
        WindowDisplayer(thInstance, hPrevInstance, szCmdLine, iCmdShow, twidth, theight, twindowName) {}
    ~RdxCuda() {}

protected:
    virtual RENDER_STATUS render(RENDER_COMMAND* renderCommand) override;
};
