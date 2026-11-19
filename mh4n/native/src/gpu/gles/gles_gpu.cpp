// The `gles` GPU (gles_gpu.h).
#include "gpu/gles/gles_gpu.h"

#include <string>

#include "gpu/gles/gles_context.h"
#include "gpu/gles/gles_render.h"
#include "gpu/pica/regs.h"
#include "mh4n/log.h"
#include "mh4n/options.h"
#include "mh4n/registry.h"

namespace mh4n::gpu::gles {

namespace {
constexpr const char* kLog = "GlesGpu";
constexpr u32 kVaddrLcd = 0x1ED02000;
}  // namespace

GlesGpu::GlesGpu(Host& host) : host_(host), gx_(host), drawer_(host), processor_(host, drawer_) {
    // The context is made on this thread (the game's: the renderer's thread, when [Graphics] RenderThread is on).
    std::string error;
    has_context_ = MakeCurrent(&error);
    if (has_context_) {
        MH4N_LOG(Info, kLog, "the GLES renderer's context: {}", Describe());
    } else {
        MH4N_LOG(Error, kLog, "the GLES renderer has no context ({}): the GX commands run, nothing is drawn", error);
    }
}

u32 GlesGpu::ToPhysical(u32 vaddr) const {
    if (vaddr == 0) {
        return 0;
    }
    return host_.VirtualToPhysical(vaddr).value_or(vaddr);
}

void GlesGpu::Execute(const GxCommand& command) {
    const u32 id = command.words[0] & 0xFF;
    if (id == static_cast<u32>(gx::CommandId::SubmitCmdList)) {
        // GPU::VirtualToPhysicalAddress: 0 stays 0, and an address in none of the windows is taken to be physical.
        processor_.SubmitGxCmdList(ToPhysical(command.words[1]), command.words[2]);
        return;
    }
    if (id == static_cast<u32>(gx::CommandId::DisplayTransfer) && !told_transfer_) {
        // What the transfer reads (the buffer the write-back has to fill) and writes, and how: a black screen says
        // nothing about where the game looks for its scene buffer.
        told_transfer_ = true;
        const u32 in_w = command.words[3] & 0xFFFF, in_h = command.words[3] >> 16;
        const u32 out_w = command.words[4] & 0xFFFF, out_h = command.words[4] >> 16;
        const u32 flags = command.words[5];
        MH4N_LOG(Info, kLog,
                 "the first display transfer: {}x{} format {} at {:#010x} -> {}x{} format {} at {:#010x}; flags {:#x} "
                 "(flip {}, input linear {}, crop {}, texture copy {}, no swizzle {}, scaling {}); the draws' last "
                 "buffer {:#010x}",
                 in_w, in_h, (flags >> 8) & 7, ToPhysical(command.words[1]), out_w, out_h, (flags >> 12) & 7,
                 ToPhysical(command.words[2]), flags, flags & 1, (flags >> 1) & 1, (flags >> 2) & 1, (flags >> 3) & 1,
                 (flags >> 5) & 1, (flags >> 24) & 3, drawer_.target().address);
    }
    // The engine's blitter (a fill, a display transfer, a texture copy) reads and writes guest memory: what the draws
    // made has to be in it first. This is how a frame reaches the screens — the game transfers its scene buffer into
    // the LCD framebuffer, which display_mem then reads.
    const bool blit = id == static_cast<u32>(gx::CommandId::MemoryFill) ||
                      id == static_cast<u32>(gx::CommandId::DisplayTransfer) ||
                      id == static_cast<u32>(gx::CommandId::TextureCopy) ||
                      id == static_cast<u32>(gx::CommandId::RequestDma);
    if (blit) {
        WriteTargetsToGuest();
    }
    gx_.Execute(command);
    if (blit) {
        // The blit wrote memory: every buffer's pass is over, so the next draw into one loads what memory holds now.
        render::EndPasses();
    }
    if (id == static_cast<u32>(gx::CommandId::CacheFlush) && command.words[2] != 0) {
        // The game told the GPU the CPU wrote the range the command names: the textures there are looked at again.
        drawer_.OnFlush(ToPhysical(command.words[1]), command.words[2]);
    }
}

void GlesGpu::WriteReg(u32 addr, u32 value) {
    gx_.WriteReg(addr, value);
    // The internal (drawing) registers are the command processor's state: the same address, past the external block.
    const u32 window = addr & 0xFFFFF000;
    if (window == pica::kGpuMmioBase || window == pica::kGpuMmioBase + 0x1000) {
        const u32 index = (addr - pica::kGpuMmioBase) / 4;
        if (index >= pica::kExternalRegs && index - pica::kExternalRegs < pica::kNumRegs) {
            processor_.WriteHardwareRegister(index - pica::kExternalRegs, value);
        }
    } else if (window != kVaddrLcd) {
        MH4N_LOG(Error, kLog, "write to unknown GPU address {:#010x}", addr);
    }
}

u32 GlesGpu::ReadReg(u32 addr) {
    const u32 window = addr & 0xFFFFF000;
    if (window == pica::kGpuMmioBase || window == pica::kGpuMmioBase + 0x1000) {
        const u32 index = (addr - pica::kGpuMmioBase) / 4;
        if (index >= pica::kExternalRegs && index - pica::kExternalRegs < pica::kNumRegs) {
            return processor_.state().regs[index - pica::kExternalRegs];
        }
    }
    return gx_.ReadReg(addr);
}

void GlesGpu::SetBufferSwap(u32 screen, const FrameBufferInfo& info) {
    if (screen < 2) {
        swap_[screen] = info;
        has_swap_[screen] = true;
        if (!told_swap_[screen]) {
            // (what the display will read this screen from, beside what the draws made: the two must agree)
            told_swap_[screen] = true;
            const render::Target& target = drawer_.target();
            MH4N_LOG(Info, kLog, "screen {} is swapped into {:#010x} (stride {}, format {}); the draws made {}x{} at {:#010x}",
                     screen, ToPhysical(info.address_left), info.stride, info.format, target.width, target.height,
                     target.address);
        }
        // The frame is drawn: the pass is over, so the next draw into that buffer starts from what guest memory holds
        // then (the display transfer put the picture there, a memory fill may have changed it).
        render::EndFrame(ToPhysical(info.address_left));
    }
    gx_.SetBufferSwap(screen, info);
}

void GlesGpu::WriteTargetsToGuest() {
    if (!has_context_) {
        return;
    }
    const u64 written = drawer_.WriteBackDrawn();
    if (written == 0) {
        return;
    }
    if (++writebacks_ == 1) {
        const render::Target& target = drawer_.target();
        MH4N_LOG(Info, kLog, "the picture is written back into guest memory ({} colour buffer(s); the last {}x{} at "
                             "{:#010x})",
                 written, target.width, target.height, target.address);
        // What the first frame drew and how: the draws' own vertex stages on the GPU, and the CPU's for the rest.
        const DrawStats& stats = drawer_.GetStats();
        MH4N_LOG(Info, kLog, "{} draws so far: {} on the GPU's vertex stage, {} on the CPU's, {} refused by it, {} "
                             "triangles; {}",
                 stats.draws, stats.gpu_draws, stats.cpu_draws, stats.gpu_failures, stats.triangles,
                 render::Describe());
    }
}

void GlesGpu::SetColorFill(u32 fill) {
    gx_.SetColorFill(fill);
}

void GlesGpu::FlushRegion(u32 vaddr, u32 size) {
    if (size != 0) {
        drawer_.OnFlush(ToPhysical(vaddr), size);
    }
}

void GlesGpu::InvalidateRegion(u32 vaddr, u32 size) {
    if (size != 0) {
        drawer_.OnInvalidate(ToPhysical(vaddr), size);
    }
}

void GlesGpu::VBlank(u32 screen) {
    gx_.VBlank(screen);
    if (screen != 0) {
        return;
    }
    // A heartbeat: even when no picture reaches the screen, the log says whether draws arrive and what they made.
    constexpr u64 kEvery = 120;
    if (++vblanks_ % kEvery != 0) {
        return;
    }
    const DrawStats& stats = drawer_.GetStats();
    const render::Target& target = drawer_.target();
    MH4N_LOG(Info, kLog, "{} vblanks: {} draws ({} on the GPU's vertex stage, {} on the CPU's, {} refused by it), {} "
                         "shadow, {} gas, {} dropped, {} failed, {} write-backs ({:#010x} {:#010x}); the last target "
                         "{}x{} at {:#010x}; {}",
             vblanks_, stats.draws, stats.gpu_draws, stats.cpu_draws, stats.gpu_failures, stats.shadow, stats.gas,
             stats.dropped, stats.failed, writebacks_, ToPhysical(swap_[0].address_left),
             ToPhysical(swap_[1].address_left), target.width, target.height, target.address, render::Describe());
}

MH4N_REGISTER(GpuFactory, "gles", [](Host& host, const Options&) -> std::unique_ptr<Gpu> {
    return std::make_unique<GlesGpu>(host);
});

}  // namespace mh4n::gpu::gles
