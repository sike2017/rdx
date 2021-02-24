#include "rdx_cuda.h"

RdxCuda::RENDER_STATUS RdxCuda::render(RENDER_COMMAND* renderCommand) {
    ra::RenderCuda renderCuda;
    size_t size = width() * height();
    uint32_t* pixels = new uint32_t[size];
    renderCuda.render(width(), height(), pixels);
    for (int index = 0; index < size; index++) {
        uint8_t* p = reinterpret_cast<uint8_t*>(pixels + index);
        setPixel(index % width(), index / width(), RGBA(p[0], p[1], p[2]));
    }
    update();
    return RENDER_STATUS::CALL_STOP_SAVE_IMAGE;
}










