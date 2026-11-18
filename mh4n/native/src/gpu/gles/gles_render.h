// The GLES rasterizer of the alternative renderer (gpu/gles): the triangles the CPU has already placed, drawn by the
// GPU's pixels' worth, as lavilao/pocket-native's raster/gles does it.
//
// Two vertex stages draw with it:
//
//   the batch      the CPU vertex unit's triangles (triangle_batch.h): positions in clip space already, drawn by a
//                  trivial stage that passes them through (gles_draw.h's recipe for the draws it cannot accelerate)
//   the draw's own a draw Azahar accelerates keeps its own PICA vertex program, translated to GLSL ES 3.10 and run by
//                  the phone's GPU over the draw's vertex buffers and index list (DrawVertexProgram, gles_shader.h):
//                  the transform, the vertex colours, the texture coordinates and the quaternions are the GPU's work,
//                  as lavilao/pocket-native's raster/gles does it
//
// Both hand the fragment stage the colour and the three texture coordinates. The fragment stage is the PICA's own: the six combiner
// (TEV) stages, read from uniforms, over the primary colour, the three texture units, the previous stage's output, the
// combiner buffer and the stages' constant colours; the alpha test; the fog, from the table the game filled; and the
// logic op (drawn into a source texture and composited, GLES having none) and the procedural texture, whose tables the
// game wrote (decoded into one float texture, as lavilao/pocket-native's proctex.rs does). What is not here yet: the
// shadow stages (the shadow map is not drawn, so a shadow texture reads white) and cube maps.
//
// Android only: on another host every call returns false with the reason, so the PC runner and the tests build and run
// (the GLES GPU does not exist there).
#pragma once

#include <array>
#include <string>
#include <vector>

#include "gpu/pica/state.h"
#include "gpu/vertex/vertex_loader.h"
#include "mh4n/types.h"

namespace mh4n::gpu::gles::render {

// The colour target's size in pixels and the guest address its draws name (the PICA framebuffer's colour buffer): two
// screens of one size are two targets, and each starts its pass empty.
struct Target {
    u32 width = 0;
    u32 height = 0;
    u32 address = 0;
    friend bool operator==(const Target&, const Target&) = default;
};

// A decoded texture to sample from: `key` identifies its pixels, so an image already uploaded for that key is drawn
// again rather than uploaded again (the texture cache's generation, see gles_draw.cpp). `wrap_s` and `wrap_t` are the
// PICA's wrap modes (0 clamp, 2 repeat, 3 mirrored repeat); `coordinate` is which of the batch's three texture
// coordinates the unit reads (0-2).
struct Texture {
    const u8* rgba8 = nullptr;  // width * height * 4, row 0 first
    u32 width = 0;
    u32 height = 0;
    u64 key = 0;
    u32 wrap_s = 2;
    u32 wrap_t = 2;
    u32 coordinate = 0;
    // The unit's filters (register bits 1 and 2): linear where the game asked for it, else nearest.
    bool mag_linear = false;
    bool min_linear = false;
};

// One combiner (TEV) stage, as its four register words hold it (gpu/fragment/regs.h): the colour sources in bits 0-11,
// the alpha sources in bits 16-27; the colour modifiers in bits 0-11, the alpha modifiers in bits 12-22; the colour
// operation in bits 0-3, the alpha operation in bits 16-19; the colour scale in bits 0-1, the alpha scale in bits 16-17.
// `constant` is the stage's constant colour as an RGBA8 word.
struct CombinerStage {
    u32 sources = 0;
    u32 modifiers = 0;
    u32 ops = 0;
    u32 scales = 0;
    u32 constant = 0;
};

// The procedural texture (the PICA's texture unit 3), as registers 0xA8-0xAD and the five tables the game wrote (their
// raw words: the renderer decodes them into one float texture, as lavilao/pocket-native's proctex.rs does).
struct ProcTexState {
    bool enable = false;
    u32 config = 0;
    u32 noise[2] = {0, 0};
    u32 frequency = 0;
    u32 lut = 0;
    u32 offset = 0;
    u32 coordinate = 0;  // which of the batch's coordinates it reads (main config bits 8-9)
    const std::array<u32, 128>* noise_map = nullptr;
    const std::array<u32, 128>* color_map = nullptr;
    const std::array<u32, 128>* alpha_map = nullptr;
    const std::array<u32, 256>* color_table = nullptr;
    const std::array<u32, 256>* color_diff_table = nullptr;
};

// The fragment half of one draw: the combiner and the alpha test. Unset, the stages pass the primary colour through,
// which is what a draw with no texture looks like.
struct FragmentState {
    CombinerStage stages[6]{};
    u32 combiner_buffer_colour = 0;  // the colour the first stage's combiner buffer starts with (RGBA8 word)
    u32 update_rgb = 0;              // stages 0-3 whose output the combiner buffer takes, bit per stage
    u32 update_alpha = 0;
    bool alpha_test = false;
    u32 alpha_func = 1;  // PICA CompareFunc (1 = Always)
    u32 alpha_ref = 0;
    // The PICA's scissor mode (Disabled 0, Exclude 1, Include 3) and, for Exclude, the register rectangle in the
    // fragment stage's pixels (x0, y0, x1, y1, y0 the bottom row): the pixels inside it are discarded.
    u32 scissor_mode = 0;
    int scissor_x0 = 0, scissor_y0 = 0, scissor_x1 = 0, scissor_y1 = 0;

    // The fog (registers 0xE0 and 0xE1): the mode the register names (5 = Fog), which way its depth runs (bit 16), the
    // colour, and the 128-entry look-up table the game filled (two half-precision words an entry, r then g: the factor is
    // r + g * fraction, as the fog code of gpu/fragment/fog/fog.cpp reads it). A null table leaves the fog off.
    u32 fog_mode = 0;
    bool fog_flip = false;
    float fog_colour[3] = {0.0f, 0.0f, 0.0f};
    const u32* fog_lut = nullptr;
    u32 fog_lut_key = 0;  // changes when the table's entries did; 0 means "not uploaded"

    ProcTexState proctex;
};

// What one draw sets before it: the viewport and scissor rectangles the registers name (in the target's pixels, row 0
// the top), the output merger's state, the three texture units, and the fragment state above.
struct DrawState {
    float viewport_x = 0.0f, viewport_y = 0.0f, viewport_width = 0.0f, viewport_height = 0.0f;
    int scissor_x = 0, scissor_y = 0;
    u32 scissor_width = 0, scissor_height = 0;

    // Output merger: the depth test and its write, the comparison (a PICA CompareFunc), the colour write mask, blending
    // with its two equations and four factors (PICA enumerators), the constant colour, the logic op (PICA), and the
    // culling the rasterizer's registers ask for (0 none, 1 back, 2 front, 3 both: as Azahar maps them, with `flipped`
    // for a framebuffer whose flip bit is clear).
    bool depth_test = true, depth_write = true;
    u32 depth_func = 4;  // LessThan
    u8 colour_write = 0xF;
    bool blend = false;
    u32 blend_equation_rgb = 0, blend_equation_alpha = 0;
    u32 blend_source_rgb = 1, blend_dest_rgb = 0;
    u32 blend_source_alpha = 1, blend_dest_alpha = 0;
    float blend_constant[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // The logic op (register 0x102, only while blending is off): Azahar's LogicOp, 3 (Copy) when there is none. GLES has
    // no logic op, so the others are drawn into a source texture and composited onto the target, bit by bit.
    u32 logic_op = 3;
    u32 cull = 0;
    bool flipped = false;

    // The draw's textures, as the texture cache decodes them; `texture_count` of them are set.
    Texture textures[3];
    int texture_count = 0;

    FragmentState fragment;
};

// The target's existing picture, as RGBA8 (width * height * 4, row 0 first): the guest memory the framebuffer names. A
// pass starts from it, as a GPU's framebuffer load does, so the game's memory fills and what was drawn into the buffer
// before its swap survive the pass. Null means an empty target (cleared to opaque black).
struct TargetContents {
    const u8* rgba8 = nullptr;
    u32 width = 0;
    u32 height = 0;
};

// True while the target's pass is going on (a draw into it already started it this frame): its picture is loaded and its
// depth cleared once, at the first draw, not at every draw.
bool IsPassStarted(const Target& target);

// Makes the renderer's objects, or makes (and starts) the colour target's pass: its existing picture is loaded from
// `contents`, and its depth is cleared. False, with `error`, on a host without GLES and on a driver that refused the
// program or the target; the reason is logged once.
bool Begin(const Target& target, const TargetContents* contents, std::string* error);

// Draws `count` vertices of the batch (triangle_batch.h's HardwareVertex layout, 88 bytes each) in triangles.
bool DrawTriangles(const void* vertices, u32 count, const DrawState& state, std::string* error);

// A draw whose vertex stage is the PICA's own program, translated to GLSL ES and run by the phone's GPU over the draw's
// vertex buffers, as lavilao/pocket-native's raster/gles does (instead of the CPU vertex unit's triangles).
struct VertexDraw {
    const pica::PicaState* state = nullptr;      // the processor's state at the draw: its vertex program and registers
    const vertex::LoadedDraw* loaded = nullptr;  // the draw's attributes, vertex buffers and indices (vertex_loader.h)
};

// Draws `draw` with the fragment state `state`. False, with `error`, when this draw cannot go that way — a program the
// translator or the driver refused, a topology or an attribute this has not — in which case the caller takes the CPU
// path, which always draws it.
bool DrawVertexProgram(const VertexDraw& draw, const DrawState& state, std::string* error);

// The screen's frame is over (the game swapped the colour buffer at `address`): that buffer's pass ends, so the next
// draw into it starts from the picture guest memory holds then.
void EndFrame(u32 address);

// The colour target as RGBA8 (`out` = width * height * 4, row 0 first): the target whose colour buffer is `address`
// (the buffer the game swapped), else the one of the last draw.
bool ReadTarget(u32 address, std::vector<u8>& out, u32* width, u32* height, std::string* error);

// One line for log.txt: the renderer's objects and the last target.
std::string Describe();

// Forgets everything the renderer holds (the driver gave the context back, or the app came back from the background
// with a new one): the next Begin makes the objects again.
void Forget();

}  // namespace mh4n::gpu::gles::render
