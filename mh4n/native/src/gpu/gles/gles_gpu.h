// The `gles` GPU: the alternative renderer, on an OpenGL ES 3.1 context of its own instead of the Vulkan one
// (gpu/native). Some drivers (the PowerVR Rogue GE8320 of a Helio G25, for one) misdraw the PICA's draws through their
// Vulkan driver; this path is meant to draw them through GLES, the way lavilao/pocket-native's raster/gles does: the
// CPU places the triangles and a GLES fragment program fills the pixels.
//
// What is here now is the path itself: the context (gles_context.h, an EGL display, no window, ES 3.1 for the storage
// buffers and the compute passes the renderer's shaders use), and the null GPU's command processing behind it, so that
// `--gpu gles-display` boots the game, signals the interrupts Azahar's GPU signals and shows what guest memory holds
// while the drawing is being built. Registered as "gles"; display_mem wraps it as "gles-display".
#pragma once

#include "gpu_null/null_gpu.h"
#include "mh4n/gpu.h"

namespace mh4n::gpu::gles {

class GlesGpu final : public Gpu {
public:
    explicit GlesGpu(Host& host);

    void Execute(const GxCommand& command) override;
    void WriteReg(u32 addr, u32 value) override;
    u32 ReadReg(u32 addr) override;
    void SetBufferSwap(u32 screen, const FrameBufferInfo& info) override;
    void SetColorFill(u32 fill) override;
    void FlushRegion(u32 vaddr, u32 size) override;
    void InvalidateRegion(u32 vaddr, u32 size) override;
    void VBlank(u32 screen) override;

    // Whether the driver gave the renderer a GLES context (the log says why when it did not).
    bool HasContext() const { return has_context_; }
    // What the GPU was asked to do, as the null GPU counts it (the drawing counts its own).
    const null::Stats& GetStats() const { return inner_.GetStats(); }

private:
    // The command processing behind the GLES path: every GX command, the register file, the fills, the DMA and the
    // interrupts. The drawing will take its registers and guest memory from here.
    null::NullGpu inner_;
    bool has_context_ = false;
};

}  // namespace mh4n::gpu::gles
