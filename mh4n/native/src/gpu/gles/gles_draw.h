// The PICA's draws on the GLES rasterizer (gpu/gles): the CPU transform and shading this port already has (N2-36's
// runner, N2-37's assembler and batch, gpu/cpuvertex), then the triangles drawn by gles_render.h.
//
// Per draw, as triangle_batch.h's recipe says:
//   1. the primitive assembler applies the draw's register side effects;
//   2. a draw Azahar would accelerate is not drawn here yet (it needs the vertex stage the Vulkan path translates), so
//      it is noted and skipped, and so is a draw in the shadow mode (it renders into no attachment);
//   3. the CPU runner shades it, the assembler builds its triangles, the batch fixes the quaternions up;
//   4. the framebuffer registers give the target (size and colour buffer), the viewport and the scissor; the texturing
//      registers the three units and the six combiner stages, the fragment registers the alpha test, and the framebuffer
//      registers the output merger (depth test and its function and write, the colour write mask, blending, culling);
//      then the rasterizer draws them.
// What is not here yet: the fog, the lighting (a draw's primary colour is its vertex colour), the procedural texture,
// the shadow stages, the logic op (GLES has none: a draw that asks for one is copied), and the accelerated draws. The
// picture is therefore the game's geometry with the PICA's own combine of its colours and textures, which is what the
// game's 2D screens are made of.
//
// Thread: the GPU's, as the command processor's (principle 6).
#pragma once

#include <string>
#include <vector>

#include "gpu/cpuvertex/cpu_vertex.h"
#include "gpu/cpuvertex/primitive_assembly.h"
#include "gpu/cpuvertex/triangle_batch.h"
#include "gpu/gles/gles_render.h"
#include "gpu/pica/backend.h"
#include "gpu/texture/cache.h"
#include "gpu/texture/texture_regs.h"
#include "mh4n/gpu.h"

namespace mh4n::gpu::gles {

struct DrawStats {
    u64 draws = 0;              // draws the PICA issued
    u64 cpu_draws = 0;          // of them, shaded on the CPU and put through the rasterizer
    u64 accelerated = 0;        // of them, ones Azahar accelerates: not drawn here yet
    u64 shadow = 0;             // of them, ones in the shadow mode (no attachment): not drawn here
    u64 dropped = 0;            // of them, ones the CPU runner could not run
    u64 failed = 0;             // of them, ones the rasterizer refused
    u64 triangles = 0;          // triangles drawn
    u64 readbacks = 0;          // pictures written into guest memory
};

// The colour formats of the PICA's framebuffer (ColorFormat; the same numbering as texture::Format's five).
u32 PicaBytesPerPixel(u32 format);
// One row of RGBA8 pixels (width * 4 bytes) as `format` writes them (`width` * PicaBytesPerPixel bytes).
void EncodeColourRow(const u8* rgba8, u32 width, u32 format, u8* out);

class Drawer final : public pica::Backend {
public:
    explicit Drawer(gpu::Host& host) : host_(host), textures_(host) {}

    // pica::Backend: one draw of the command processor's.
    void Draw(const pica::DrawCall& draw) override;

    // The CPU wrote, or the GPU wrote, [paddr, paddr + size): the textures it touches are looked at again.
    void OnFlush(u32 paddr, u32 size) {
        textures_.OnFlush(paddr, size);
    }
    void OnInvalidate(u32 paddr, u32 size) {
        textures_.OnInvalidate(paddr, size);
    }

    // The last pass's colour target, written into guest memory as the frame buffer `info` names: `address` for `stride`
    // bytes a row, of `format`, for `height` rows. False when there is no picture or the memory does not serve it.
    bool WriteColourToGuest(u32 address, u32 stride, u32 height, u32 format);

    // The picture's size (the framebuffer of the last draw), 0 when nothing was drawn.
    const render::Target& target() const {
        return target_;
    }
    const DrawStats& GetStats() const {
        return stats_;
    }

private:
    gpu::Host& host_;
    cpuvertex::CpuVertexRunner runner_;
    cpuvertex::PrimitiveAssembler assembler_;
    cpuvertex::TriangleBatch batch_;
    texture::TextureCache textures_;
    std::vector<texture::TextureRef> texture_refs_;
    render::Target target_;
    std::vector<u8> picture_;  // the last read back picture (RGBA8)
    DrawStats stats_;
    bool told_failure_ = false;
};

}  // namespace mh4n::gpu::gles
