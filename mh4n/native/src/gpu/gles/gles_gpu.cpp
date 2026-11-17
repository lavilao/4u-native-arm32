// The `gles` GPU (gles_gpu.h).
#include "gpu/gles/gles_gpu.h"

#include <string>

#include "gpu/gles/gles_context.h"
#include "mh4n/log.h"
#include "mh4n/options.h"
#include "mh4n/registry.h"

namespace mh4n::gpu::gles {

namespace {
constexpr const char* kLog = "GlesGpu";
}  // namespace

GlesGpu::GlesGpu(Host& host) : inner_(host) {
    // The context is made on this thread (the game's: the runtime makes its GPU there); a renderer thread of its own
    // makes it current on that one when the drawing lands.
    std::string error;
    has_context_ = MakeCurrent(&error);
    if (has_context_) {
        MH4N_LOG(Info, kLog, "the GLES renderer's context: {}", Describe());
    } else {
        MH4N_LOG(Error, kLog,
                 "the GLES renderer has no context ({}): the GX commands run as the null GPU's, nothing is drawn",
                 error);
    }
}

void GlesGpu::Execute(const GxCommand& command) {
    inner_.Execute(command);
}

void GlesGpu::WriteReg(u32 addr, u32 value) {
    inner_.WriteReg(addr, value);
}

u32 GlesGpu::ReadReg(u32 addr) {
    return inner_.ReadReg(addr);
}

void GlesGpu::SetBufferSwap(u32 screen, const FrameBufferInfo& info) {
    inner_.SetBufferSwap(screen, info);
}

void GlesGpu::SetColorFill(u32 fill) {
    inner_.SetColorFill(fill);
}

void GlesGpu::FlushRegion(u32 vaddr, u32 size) {
    inner_.FlushRegion(vaddr, size);
}

void GlesGpu::InvalidateRegion(u32 vaddr, u32 size) {
    inner_.InvalidateRegion(vaddr, size);
}

void GlesGpu::VBlank(u32 screen) {
    inner_.VBlank(screen);
}

MH4N_REGISTER(GpuFactory, "gles", [](Host& host, const Options&) -> std::unique_ptr<Gpu> {
    return std::make_unique<GlesGpu>(host);
});

}  // namespace mh4n::gpu::gles
