// The GLES rasterizer of the alternative renderer (gpu/gles): the triangles the CPU has already placed, drawn by the
// GPU's pixels' worth, as lavilao/pocket-native's raster/gles does it.
//
// The one program here is the PICA's simplest case: the vertex stage only passes the batch through (its positions are
// clip space already, node 2 of triangle_batch.h's recipe), and the fragment stage gives the vertex colour, times the
// first texture when the draw has one. The texture combiner, the fog, the alpha test and the output merger come after;
// what is drawn now is what a draw looks like without them.
//
// Android only: on another host every call returns false with the reason, so the PC runner and the tests build and run
// (the GLES GPU does not exist there).
#pragma once

#include <string>
#include <vector>

#include "mh4n/types.h"

namespace mh4n::gpu::gles::render {

// The colour target's size in pixels (the PICA framebuffer the draws go into).
struct Target {
    u32 width = 0;
    u32 height = 0;
    friend bool operator==(const Target&, const Target&) = default;
};

// What one draw sets before it: the viewport and scissor rectangles the registers name (in the target's pixels, row 0
// the top) and the depth test the draw asks for.
struct DrawState {
    float viewport_x = 0.0f, viewport_y = 0.0f, viewport_width = 0.0f, viewport_height = 0.0f;
    int scissor_x = 0, scissor_y = 0;
    u32 scissor_width = 0, scissor_height = 0;
    bool depth_test = true, depth_write = true;
};

// A decoded texture to sample from: `key` identifies its pixels, so an image already uploaded for that key is drawn
// again rather than uploaded again (the texture cache's generation, see gles_draw.cpp).
struct Texture {
    const u8* rgba8 = nullptr;  // width * height * 4, row 0 first
    u32 width = 0;
    u32 height = 0;
    u64 key = 0;
};

// Makes the renderer's objects, or resizes the colour target and clears it: the picture of a pass starts empty, as the
// game's first draw of it expects (it fills it). False, with `error`, on a host without GLES and on a driver that
// refused the program or the target; the reason is logged once.
bool Begin(const Target& target, std::string* error);

// Draws `count` vertices of the batch (triangle_batch.h's HardwareVertex layout, 88 bytes each) in triangles.
bool DrawTriangles(const void* vertices, u32 count, const DrawState& state, const Texture* texture, std::string* error);

// The colour target as RGBA8 (`out` = width * height * 4, row 0 first).
bool ReadTarget(std::vector<u8>& out, u32* width, u32* height, std::string* error);

// One line for log.txt: the renderer's objects and the last target.
std::string Describe();

// Forgets everything the renderer holds (the driver gave the context back, or the app came back from the background
// with a new one): the next Begin makes the objects again.
void Forget();

}  // namespace mh4n::gpu::gles::render
