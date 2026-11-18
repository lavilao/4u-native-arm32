// The PICA's draws on the GLES rasterizer (gpu/gles): the CPU transform and shading this port already has (N2-36's
// runner, N2-37's assembler and batch, gpu/cpuvertex), then the triangles drawn by gles_render.h.
//
// Per draw, as triangle_batch.h's recipe says:
//   1. the primitive assembler applies the draw's register side effects;
//   2. a draw in the shadow mode (it renders into the shadow map) or of the fog's density pass is skipped: neither has a
//      colour; a draw Azahar accelerates keeps its own PICA vertex and fragment programs: see 3;
//   3. a draw Azahar accelerates is loaded (vertex_loader.h) and its own vertex program, translated to GLSL ES, runs on
//      the GPU over the draw's buffers (gles_render.h DrawVertexProgram), as lavilao/pocket-native's raster/gles does;
//      a draw that cannot go that way — the CPU path's own, a program the translator or the driver refused, a topology
//      this has not — is shaded by the CPU vertex unit, the assembler builds its triangles and the batch fixes the
//      quaternions up;
//   4. the framebuffer registers give the target (size and colour buffer), the viewport and the scissor; the texturing
//      registers the three units and the six combiner stages, the fragment registers the alpha test, and the framebuffer
//      registers the output merger (depth test and its function and write, the colour write mask, blending, culling);
//      then the rasterizer draws them.
// The PICA's fog (mode 5) is applied after the combiner, from the table the game filled; the procedural texture comes
// from its own tables; lighting is the vertex unit's (the PICA shades per vertex in its vertex shader, so the primary
// colour is the lit vertex colour). What is not here yet: the shadow stages and cube maps.
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
    u64 gpu_draws = 0;          // of them, whose own PICA program the GPU's vertex stage ran (DrawVertexProgram)
    u64 gpu_failures = 0;       // of them, ones the GPU vertex stage could not take (they went the CPU way)
    u64 accelerated = 0;        // of them, ones Azahar accelerates (the GPU vertex stage, or the CPU's when it cannot)
    u64 shadow = 0;             // of them, ones in the shadow mode (no attachment): not drawn here
    u64 gas = 0;                // of them, ones of the fog's density pass (no colour): not drawn here
    u64 dropped = 0;            // of them, ones the CPU runner could not run
    u64 failed = 0;             // of them, ones the rasterizer refused
    u64 triangles = 0;          // triangles drawn
    u64 readbacks = 0;          // colour buffers written into guest memory
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

    // Every colour buffer drawn into since the last call, written into guest memory at its own address, in the PICA's
    // 8x8 tiled layout and its colour format (gles_render.h WriteBackDrawn). The engine's display transfer reads its
    // source from there, so this is what puts a frame on the screens.
    u64 WriteBackDrawn();

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
    vertex::VertexLoader loader_{vertex::VertexCaps{}};  // the draw's attributes and buffers, for the GPU vertex stage
    vertex::LoadedDraw loaded_;
    cpuvertex::PrimitiveAssembler assembler_;
    cpuvertex::TriangleBatch batch_;
    texture::TextureCache textures_;
    std::vector<texture::TextureRef> texture_refs_;
    render::Target target_;
    std::vector<u8> picture_;  // the last read back picture (RGBA8)
    std::vector<u8> seed_;     // the colour buffer as guest memory holds it, for a pass's first draw (RGBA8)
    DrawStats stats_;
    bool told_failure_ = false;
    bool told_gpu_failure_ = false;  // (the GPU vertex stage's first refusal, logged once)
    bool told_first_draw_ = false;   // (the first draw's target, viewport and state, logged once)
    bool told_writeback_ = false;    // (why a write-back failed, logged once)
};

}  // namespace mh4n::gpu::gles
