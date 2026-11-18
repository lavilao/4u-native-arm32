// The `gles` GPU: the alternative renderer, on an OpenGL ES 3.1 context of its own instead of the Vulkan one
// (gpu/native). Some drivers (the PowerVR Rogue GE8320 of a Helio G25, for one) misdraw the PICA's draws through their
// Vulkan driver; this path draws them through GLES, the way lavilao/pocket-native's raster/gles does: the CPU places
// and shades the triangles, and a GLES program fills the pixels (gles_render.h).
//
// What it is made of, as native_gpu.cpp is:
//   - the GX engine (gpu/gx), for the DMA, the fills, the display transfers and the texture copies, the register file
//     and the interrupts;
//   - the PICA command processor (gpu/pica), which decodes the command lists into draws;
//   - the drawer (gles_draw.h), which puts each draw through the CPU vertex path and the GLES rasterizer.
// The game never draws into the LCD buffers display_mem reads: it draws its scene into a colour buffer and asks the GX
// engine for a display transfer from it into the screen's framebuffer. So before every fill, transfer and texture copy
// the pictures the draws made are written back into guest memory, at their own colour buffers, in the PICA's tiled
// layout; the engine's blitter then reads what the GLES rasterizer drew.
//
// Registered as "gles"; display_mem wraps it as "gles-display", which the app picks with [Graphics] Renderer = gles (or
// the row in the options) in mh4u-app.ini.
#pragma once

#include "gpu/gles/gles_draw.h"
#include "gpu/gx/engine.h"
#include "gpu/pica/command_processor.h"
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
    bool HasContext() const {
        return has_context_;
    }
    const DrawStats& GetDrawStats() const {
        return drawer_.GetStats();
    }

private:
    // The pictures the draws made since the last call, into their colour buffers in guest memory (before a command that
    // reads or writes memory).
    void WriteTargetsToGuest();
    // A guest virtual address as a physical one (the texture cache's and the framebuffer's are physical).
    u32 ToPhysical(u32 vaddr) const;

    Host& host_;
    gx::GxEngine gx_;
    Drawer drawer_;
    pica::CommandProcessor processor_;
    FrameBufferInfo swap_[2] = {};
    bool has_swap_[2] = {false, false};
    bool has_context_ = false;
    bool told_swap_[2] = {false, false};  // the first swap of each screen, with its framebuffer registers
    u64 writebacks_ = 0;
    u64 vblanks_ = 0;  // top-screen VBlanks (the heartbeat's clock)
};

}  // namespace mh4n::gpu::gles
