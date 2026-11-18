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
    gx_.Execute(command);
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
        // The frame is drawn: its picture goes where the display reads it from.
        WriteBack(screen);
        // ... and the pass is over: the next draw into that buffer starts from what memory holds (WriteBack put it
        // there, a memory fill may have changed it).
        render::EndFrame(ToPhysical(info.address_left));
    }
    gx_.SetBufferSwap(screen, info);
}

void GlesGpu::WriteBack(u32 screen) {
    if (screen >= 2 || !has_swap_[screen]) {
        return;
    }
    const render::Target& target = drawer_.target();
    if (target.width == 0 || target.height == 0) {
        return;  // nothing was drawn yet
    }
    const FrameBufferInfo& info = swap_[screen];
    const u32 address = ToPhysical(info.address_left);
    if (info.stride == 0 || address == 0) {
        return;
    }
    if (drawer_.WriteColourToGuest(address, info.stride, target.height, info.format)) {
        if (++writebacks_ == 1) {
            MH4N_LOG(Info, kLog, "the picture is written back into guest memory ({}x{}, screen {})", target.width,
                     target.height, screen);
        }
        return;
    }
    if (!told_writeback_failure_) {
        told_writeback_failure_ = true;
        MH4N_LOG(Error, kLog, "the picture could not be written back (screen {}, {:#010x}, stride {})", screen, address,
                 info.stride);
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
}

MH4N_REGISTER(GpuFactory, "gles", [](Host& host, const Options&) -> std::unique_ptr<Gpu> {
    return std::make_unique<GlesGpu>(host);
});

}  // namespace mh4n::gpu::gles
