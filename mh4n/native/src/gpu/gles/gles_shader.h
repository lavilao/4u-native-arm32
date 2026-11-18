// The vertex stage's GLSL, Vulkan's and GLES's (gpu/gles): the PICA's own vertex program is translated once, as
// shader/vertex_shader.h does for the Vulkan path (Azahar's GLSL 450 generator), and this file turns that source into a
// GLSL ES 3.10 stage that the phone's GLES driver compiles, as lavilao/pocket-native's raster/gles/sources.rs
// (vertex_es) does with its own generated stage.
//
// What the conversion changes, and why:
//
//   #version 450                    -> #version 310 es with the float and int precisions (ES has no default for either)
//   layout (set = 0, binding = N)   -> layout by index only (ES 3.0 has no binding for a uniform block; the renderer
//                                      binds the block by its index, glUniformBlockBinding)
//   gl_Position z                   -> the ES's depth: the PICA's visible z runs [-w, 0], Vulkan's NDC is [0, 1] and its
//                                      -z is the depth, the ES's NDC is [-1, 1], so z becomes -2z - w and the depth
//                                      ends [-z/w] as Vulkan's does (the rule pocket-native's vertex_es writes)
//   the seven varyings              -> the four the renderer's fragment program declares by name (v_colour,
//                                      v_texcoord0..2): GLES matches them by name, so the stage and the program link
//                                      without locations, as ES 3.0 needs. texcoord0_w, normquat and view go (that
//                                      fragment program does not light: see gles_render.h)
//   gl_ClipDistance                 -> nothing: GLSL ES has no gl_ClipDistance (the Vulkan stage is generated with
//                                      use_clip_planes false for this renderer, and this checks it did), so the near
//                                      plane is the hardware's alone
//
// A source with a shape this does not know is refused (false and the reason), never guessed at: the draw then goes
// through the CPU vertex unit, which always draws it.
#pragma once

#include <string>

namespace mh4n::gpu::gles {

// The GLSL ES 3.10 vertex stage of `vulkan_glsl` (shader/vertex_shader.h's GenerateVertexGlsl with a config made for
// this renderer). False, with the reason in `error`, when the source is not the shape described above.
bool ToEsVertexStage(const std::string& vulkan_glsl, std::string* out, std::string* error);

}  // namespace mh4n::gpu::gles
