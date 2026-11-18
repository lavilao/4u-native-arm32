// The GLES rasterizer (gles_render.h).
#include "gpu/gles/gles_render.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>

#include "gpu/cpuvertex/triangle_batch.h"
#include "gpu/gles/gles_context.h"
#include "gpu/gles/gles_shader.h"
#include "gpu/shader/vertex_shader.h"
#include "gpu/vertex/vertex_loader.h"
#include "mh4n/log.h"

#if defined(__ANDROID__)

#include <GLES3/gl3.h>

namespace mh4n::gpu::gles::render {

namespace {

constexpr const char* kLog = "GlesGpu";

// triangle_batch.h's HardwareVertex: 88 bytes, attributes 0 position, 1 colour, 2 texcoord0, 3 texcoord1, 4 texcoord2,
// 5 texcoord0_w, 6 normquat, 7 view. The stages here read the position, the colour and the three coordinates.
constexpr unsigned kVertexStride = 88;
constexpr unsigned kAttribCount = 8;
// The images kept for the textures drawn: past this many, the least useful go (a new pass uploads them again).
constexpr std::size_t kMaxImages = 192;

// The vertex stage: the batch is clip space already (the CPU placed it), so it goes straight out, and the colour and the
// three texture coordinates go with it.
constexpr const char* kVertexSource = R"(#version 310 es
layout(location = 0) in vec4 a_position;
layout(location = 1) in vec4 a_colour;
layout(location = 2) in vec2 a_texcoord0;
layout(location = 3) in vec2 a_texcoord1;
layout(location = 4) in vec2 a_texcoord2;
uniform int u_flip_y;
out vec4 v_colour;
out vec2 v_texcoord0;
out vec2 v_texcoord1;
out vec2 v_texcoord2;
void main() {
    // As the Vulkan path's vertex stage: y is negated for a framebuffer whose flip bit is clear (`flip_viewport`), and
    // the PICA's z (visible from -w to 0) becomes the ES's depth: Vulkan's NDC is [0, 1] and its depth -z/w, the ES's
    // NDC is [-1, 1], so the stage writes -2z - w and the depth ends -z/w again (the rule pocket-native's vertex_es
    // writes for its own stage).
    gl_Position = vec4(a_position.x, u_flip_y != 0 ? -a_position.y : a_position.y,
                       -2.0 * a_position.z - a_position.w, a_position.w);
    gl_PointSize = 1.0;
    v_colour = a_colour;
    v_texcoord0 = a_texcoord0;
    v_texcoord1 = a_texcoord1;
    v_texcoord2 = a_texcoord2;
}
)";

// The fragment stage: the PICA's six combiner stages and the alpha test. The tables are the ones of
// gpu/fragment/generator.cpp (Azahar's switch statements, one entry per enumerator); the stages read their registers
// from uniforms, so one program draws every combination.
constexpr const char* kFragmentSource = R"(#version 310 es
precision highp float;

in vec4 v_colour;
in vec2 v_texcoord0;
in vec2 v_texcoord1;
in vec2 v_texcoord2;

uniform sampler2D u_texture0;
uniform sampler2D u_texture1;
uniform sampler2D u_texture2;
uniform int u_has_texture0;
uniform int u_has_texture1;
uniform int u_has_texture2;
uniform int u_coordinate0;
uniform int u_coordinate1;
uniform int u_coordinate2;

uniform int u_source[6];
uniform int u_modifier[6];
uniform int u_ops[6];
uniform int u_scale[6];
uniform vec4 u_constant[6];
uniform vec4 u_combiner_colour;
uniform int u_update_rgb;
uniform int u_update_alpha;
uniform int u_alpha_test;
uniform int u_alpha_func;
uniform int u_alpha_ref;
uniform int u_scissor_mode;
uniform ivec4 u_scissor;

uniform sampler2D u_fog_lut;
uniform int u_fog_mode;
uniform int u_fog_flip;
uniform vec3 u_fog_colour;

uniform sampler2D u_proctex;
uniform int u_proctex_enable;
uniform int u_proctex_config;
uniform int u_proctex_lut;
uniform int u_proctex_offset;
uniform int u_proctex_coordinate;
uniform vec2 u_proctex_amplitude;
uniform vec2 u_proctex_phase;
uniform vec2 u_proctex_frequency;

out vec4 o_colour;

// The PICA's eight bits of precision between stages (generator.cpp's byteround).
float byteround(float x) {
    return round(clamp(x, 0.0, 1.0) * 255.0) / 255.0;
}
vec3 byteround(vec3 x) {
    return vec3(byteround(x.x), byteround(x.y), byteround(x.z));
}
vec4 byteround(vec4 x) {
    return vec4(byteround(x.x), byteround(x.y), byteround(x.z), byteround(x.w));
}

// Which of the batch's texture coordinates a unit reads.
vec2 Coordinate(int which, vec2 c0, vec2 c1, vec2 c2) {
    if (which == 1) {
        return c1;
    }
    if (which == 2) {
        return c2;
    }
    return c0;
}

// TevStageConfig::Source: PrimaryColor 0, PrimaryFragmentColor 1, SecondaryFragmentColor 2, Texture0-2 3-5,
// Texture3 6 (the procedural texture), PreviousBuffer 0xD, Constant 0xE, Previous 0xF. The primary fragment
// colour is the lighting result, which this renderer does not shade: the vertex colour stands in for it, and the
// secondary colour is nothing.
vec4 TevSource(int source, vec4 primary, vec4 previous, vec4 held, vec4 constant_colour,
               vec4 texture0, vec4 texture1, vec4 texture2, vec4 texture3) {
    if (source == 1) {
        return primary;
    }
    if (source == 2) {
        return vec4(0.0);
    }
    if (source == 3) {
        return texture0;
    }
    if (source == 4) {
        return texture1;
    }
    if (source == 5) {
        return texture2;
    }
    if (source == 6) {
        return texture3;
    }
    if (source == 0xD) {
        return held;
    }
    if (source == 0xE) {
        return constant_colour;
    }
    if (source == 0xF) {
        return previous;
    }
    return source == 0 ? primary : vec4(0.0);
}

// TevStageConfig::ColorModifier.
vec3 ColorModifier(vec4 source, int modifier) {
    if (modifier == 0) {
        return source.rgb;
    }
    if (modifier == 1) {
        return vec3(1.0) - source.rgb;
    }
    if (modifier == 2) {
        return source.aaa;
    }
    if (modifier == 3) {
        return vec3(1.0) - source.aaa;
    }
    if (modifier == 4) {
        return source.rrr;
    }
    if (modifier == 5) {
        return vec3(1.0) - source.rrr;
    }
    if (modifier == 8) {
        return source.ggg;
    }
    if (modifier == 9) {
        return vec3(1.0) - source.ggg;
    }
    if (modifier == 0xC) {
        return source.bbb;
    }
    if (modifier == 0xD) {
        return vec3(1.0) - source.bbb;
    }
    return vec3(0.0);
}

// TevStageConfig::AlphaModifier.
float AlphaModifier(vec4 source, int modifier) {
    if (modifier == 0) {
        return source.a;
    }
    if (modifier == 1) {
        return 1.0 - source.a;
    }
    if (modifier == 2) {
        return source.r;
    }
    if (modifier == 3) {
        return 1.0 - source.r;
    }
    if (modifier == 4) {
        return source.g;
    }
    if (modifier == 5) {
        return 1.0 - source.g;
    }
    if (modifier == 6) {
        return source.b;
    }
    if (modifier == 7) {
        return 1.0 - source.b;
    }
    return 0.0;
}

// TevStageConfig::Operation, before the clamp: Replace 0, Modulate 1, Add 2, AddSigned 3, Lerp 4, Subtract 5,
// Dot3_RGB 6, Dot3_RGBA 7, MultiplyThenAdd 8, AddThenMultiply 9.
vec3 ColorCombiner(int operation, vec3 a, vec3 b, vec3 c) {
    if (operation == 0) {
        return a;
    }
    if (operation == 1) {
        return a * b;
    }
    if (operation == 2) {
        return a + b;
    }
    if (operation == 3) {
        return a + b - vec3(0.5);
    }
    if (operation == 4) {
        return mix(b, a, c);
    }
    if (operation == 5) {
        return a - b;
    }
    if (operation == 6 || operation == 7) {
        return vec3(dot(a - vec3(0.5), b - vec3(0.5)) * 4.0);
    }
    if (operation == 8) {
        return a * b + c;
    }
    if (operation == 9) {
        return min(a + b, vec3(1.0)) * c;
    }
    return vec3(0.0);
}

// The same operations on alpha; Dot3_RGB and Dot3_RGBA have none.
float AlphaCombiner(int operation, float a, float b, float c) {
    if (operation == 0) {
        return a;
    }
    if (operation == 1) {
        return a * b;
    }
    if (operation == 2) {
        return a + b;
    }
    if (operation == 3) {
        return a + b - 0.5;
    }
    if (operation == 4) {
        return mix(b, a, c);
    }
    if (operation == 5) {
        return a - b;
    }
    if (operation == 8) {
        return a * b + c;
    }
    if (operation == 9) {
        return min(a + b, 1.0) * c;
    }
    return 0.0;
}

// GetColorMultiplier / GetAlphaMultiplier: `(scale < 3) ? (1 << scale) : 1`.
float Multiplier(int scale) {
    return scale == 1 ? 2.0 : (scale == 2 ? 4.0 : 1.0);
}

// The condition under which the alpha test discards the fragment (generator.cpp's kAlphaTests).
bool AlphaTestFails(int function, int alpha) {
    if (function == 0) {
        return true;  // Never
    }
    if (function == 1) {
        return false;  // Always
    }
    if (function == 2) {
        return alpha != u_alpha_ref;
    }
    if (function == 3) {
        return alpha == u_alpha_ref;
    }
    if (function == 4) {
        return alpha >= u_alpha_ref;
    }
    if (function == 5) {
        return alpha > u_alpha_ref;
    }
    if (function == 6) {
        return alpha <= u_alpha_ref;
    }
    return alpha < u_alpha_ref;
}

// ---- the procedural texture (lavilao/pocket-native's proctex.rs, in GLSL) -----------------------------------------

// The noise table has 16 rows and 16 mixes (their random()'s two constant tables).
uint ProcTexRows(uint i) {
    if (i == 0u) return 0u;
    if (i == 1u) return 4u;
    if (i == 2u) return 10u;
    if (i == 3u) return 8u;
    if (i == 4u) return 4u;
    if (i == 5u) return 9u;
    if (i == 6u) return 7u;
    if (i == 7u) return 12u;
    if (i == 8u) return 5u;
    if (i == 9u) return 15u;
    if (i == 10u) return 13u;
    if (i == 11u) return 14u;
    if (i == 12u) return 11u;
    if (i == 13u) return 15u;
    if (i == 14u) return 2u;
    return 11u;
}
uint ProcTexMix(uint i) {
    if (i == 0u) return 10u;
    if (i == 1u) return 2u;
    if (i == 2u) return 15u;
    if (i == 3u) return 8u;
    if (i == 4u) return 0u;
    if (i == 5u) return 7u;
    if (i == 6u) return 4u;
    if (i == 7u) return 5u;
    if (i == 8u) return 5u;
    if (i == 9u) return 13u;
    if (i == 10u) return 2u;
    if (i == 11u) return 6u;
    if (i == 12u) return 13u;
    if (i == 13u) return 9u;
    if (i == 14u) return 3u;
    return 14u;
}

// Row `map` (0 noise, 1 colour map, 2 alpha map) at a coordinate from 0 to 1, between its entries (value, step).
float ProcTexLookup(int map, float coordinate) {
    float at = coordinate * 128.0;
    int entry = clamp(int(at), 0, 127);
    vec2 value = texelFetch(u_proctex, ivec2(entry, map), 0).rg;
    return value.r + (at - float(entry)) * value.g;
}

// A pseudo-random value from -1 to 1 for a point of the noise's grid.
float ProcTexRandom(int x, int y) {
    uint u = x < 0 ? 0u : uint(x);
    uint v = y < 0 ? 0u : uint(y);
    uint one_u = ((((u % 9u) + 2u) * 3u) & 0xFu) ^ ProcTexRows((u / 9u) & 0xFu);
    uint one_v = ((((v % 9u) + 2u) * 3u) & 0xFu) ^ ProcTexRows((v / 9u) & 0xFu);
    if ((one_u & 3u) == 1u) {
        one_v += 4u;
    }
    one_v ^= (one_u & 1u) * 6u;
    one_v += 10u + one_u;
    one_v &= 0xFu;
    one_v ^= ProcTexMix(one_u & 0xFu);
    return -1.0 + float(one_v) * 2.0 / 15.0;
}

// How far the shift moves one coordinate, by the other.
float ProcTexShift(float other, int mode, int edge) {
    float amount = edge == 3 ? 1.0 : 0.5;
    int whole = int(other);
    if (mode == 1) {
        return amount * float((whole / 2) % 2);
    }
    if (mode == 2) {
        return amount * float(((whole + 1) / 2) % 2);
    }
    return 0.0;
}

// Brings a coordinate back to 0 to 1: to zero, to the edge, repeat, mirrored repeat, pulse.
float ProcTexWrap(float c, int mode) {
    if (mode == 0) {
        return c > 1.0 ? 0.0 : c;
    }
    if (mode == 1) {
        return min(c, 1.0);
    }
    if (mode == 2) {
        return c - floor(c);
    }
    if (mode == 3) {
        int whole = int(c);
        float part = c - float(whole);
        return whole % 2 == 0 ? part : 1.0 - part;
    }
    if (mode == 4) {
        return c <= 0.5 ? 0.0 : 1.0;
    }
    return clamp(c, 0.0, 1.0);
}

// The two coordinates made into one (ProcTexCombiner).
float ProcTexCombine(float u, float v, int function) {
    float len = sqrt(u * u + v * v);
    if (function == 0) return u;
    if (function == 1) return u * u;
    if (function == 2) return v;
    if (function == 3) return v * v;
    if (function == 4) return (u + v) * 0.5;
    if (function == 5) return (u * u + v * v) * 0.5;
    if (function == 6) return min(len, 1.0);
    if (function == 7) return min(u, v);
    if (function == 8) return max(u, v);
    if (function == 9) return min(((u + v) * 0.5 + len) * 0.5, 1.0);
    return 0.0;
}

// The value the noise moves a pair of coordinates by, -1 to 1.
float ProcTexNoise(float u, float v) {
    float x = 9.0 * u_proctex_frequency.x * abs(u + u_proctex_phase.x);
    float y = 9.0 * u_proctex_frequency.y * abs(v + u_proctex_phase.y);
    int xi = int(x);
    int yi = int(y);
    float xf = x - float(xi);
    float yf = y - float(yi);
    float g0 = ProcTexRandom(xi, yi) * (xf + yf);
    float g1 = ProcTexRandom(xi + 1, yi) * (xf + yf - 1.0);
    float g2 = ProcTexRandom(xi, yi + 1) * (xf + yf - 1.0);
    float g3 = ProcTexRandom(xi + 1, yi + 1) * (xf + yf - 2.0);
    float s = ProcTexLookup(0, xf);
    float t = ProcTexLookup(0, yf);
    float bottom = g0 * (1.0 - s) + g1 * s;
    float top = g2 * (1.0 - s) + g3 * s;
    return bottom * (1.0 - t) + top * t;
}

// The procedural texture at a pair of coordinates, RGBA 0 to 1.
vec4 ProcTexSample(vec2 coordinate) {
    int config = u_proctex_config;
    int edge_u = config & 7;
    int edge_v = (config >> 3) & 7;
    float u = abs(coordinate.x);
    float v = abs(coordinate.y);
    // The shifts come from the coordinates before the noise moves them.
    float shift_u = ProcTexShift(v, (config >> 16) & 3, edge_u);
    float shift_v = ProcTexShift(u, (config >> 18) & 3, edge_v);
    if ((config & (1 << 15)) != 0) {
        float noise = ProcTexNoise(u, v);
        u = abs(u + noise * u_proctex_amplitude.x);
        v = abs(v + noise * u_proctex_amplitude.y);
    }
    u = ProcTexWrap(u + shift_u, edge_u);
    v = ProcTexWrap(v + shift_v, edge_v);

    float coordinate_in_table = ProcTexLookup(1, ProcTexCombine(u, v, (config >> 6) & 0xF));
    float width = float((u_proctex_lut >> 11) & 0xFF);
    float position = float(u_proctex_offset & 0xFF) + coordinate_in_table * max(width - 1.0, 0.0);
    int entry = clamp(int(position), 0, 255);
    int lut_filter = u_proctex_lut & 7;
    vec4 color;
    if (lut_filter == 1 || lut_filter == 3 || lut_filter == 5) {
        // linear (and the two linear mipmap filters): the step to the next entry is the colour table's pair
        color = (texelFetch(u_proctex, ivec2(entry, 3), 0) +
                 (position - float(entry)) * texelFetch(u_proctex, ivec2(entry, 4), 0)) / 255.0;
    } else {
        color = texelFetch(u_proctex, ivec2(clamp(int(position + 0.5), 0, 255), 3), 0) / 255.0;
    }
    color = clamp(color, vec4(0.0), vec4(1.0));
    // An alpha of its own skips the colour table: the alpha map gives it.
    if ((config & (1 << 14)) != 0) {
        color.a = clamp(ProcTexLookup(2, ProcTexCombine(u, v, (config >> 10) & 0xF)), 0.0, 1.0);
    }
    return color;
}

void main() {
    // The PICA's scissor mode Exclude: the pixels inside the register rectangle are not drawn (Include is the scissor
    // rectangle the GL state carries).
    if (u_scissor_mode == 1) {
        int x = int(gl_FragCoord.x);
        int y = int(gl_FragCoord.y);
        if (x >= u_scissor.x && x < u_scissor.z && y >= u_scissor.y && y < u_scissor.w) {
            discard;
        }
    }
    vec4 primary = byteround(v_colour);
    vec4 texture0 = u_has_texture0 != 0 ? texture(u_texture0, Coordinate(u_coordinate0, v_texcoord0, v_texcoord1, v_texcoord2)) : vec4(0.0);
    vec4 texture1 = u_has_texture1 != 0 ? texture(u_texture1, Coordinate(u_coordinate1, v_texcoord0, v_texcoord1, v_texcoord2)) : vec4(0.0);
    vec4 texture2 = u_has_texture2 != 0 ? texture(u_texture2, Coordinate(u_coordinate2, v_texcoord0, v_texcoord1, v_texcoord2)) : vec4(0.0);
    vec4 texture3 = u_proctex_enable != 0 ? ProcTexSample(Coordinate(u_proctex_coordinate, v_texcoord0, v_texcoord1, v_texcoord2)) : vec4(0.0);

    vec4 previous = vec4(0.0);            // combiner_output
    vec4 held = vec4(0.0);                // combiner_buffer
    vec4 next_buffer = u_combiner_colour; // next_combiner_buffer
    for (int stage = 0; stage < 6; ++stage) {
        int sources = u_source[stage];
        int modifiers = u_modifier[stage];
        int ops = u_ops[stage];
        int scales = u_scale[stage];
        vec4 constant_colour = u_constant[stage];
        int source[3];
        int modifier[3];
        source[0] = sources & 0xF;
        source[1] = (sources >> 4) & 0xF;
        source[2] = (sources >> 8) & 0xF;
        modifier[0] = modifiers & 0xF;
        modifier[1] = (modifiers >> 4) & 0xF;
        modifier[2] = (modifiers >> 8) & 0xF;
        // Stage 0's forces its third colour source: a Previous there is the third source register (generator.cpp's
        // ForcedSource).
        if (stage == 0) {
            for (int i = 0; i < 3; ++i) {
                if (source[i] == 0xF) {
                    source[i] = source[2];
                }
            }
        }
        vec3 colors[3];
        for (int i = 0; i < 3; ++i) {
            colors[i] = ColorModifier(TevSource(source[i], primary, previous, held, constant_colour,
                                                texture0, texture1, texture2, texture3),
                                      modifier[i]);
        }
        int color_op = ops & 0xF;
        int alpha_op = (ops >> 16) & 0xF;
        vec3 color_output = byteround(ColorCombiner(color_op, colors[0], colors[1], colors[2]));
        float alpha_output = 0.0;
        if (color_op == 7) {
            // Dot3_RGBA places its result in alpha too.
            alpha_output = color_output.r;
        } else {
            int alpha_source[3];
            int alpha_modifier[3];
            alpha_source[0] = (sources >> 16) & 0xF;
            alpha_source[1] = (sources >> 20) & 0xF;
            alpha_source[2] = (sources >> 24) & 0xF;
            alpha_modifier[0] = (modifiers >> 12) & 0x7;
            alpha_modifier[1] = (modifiers >> 16) & 0x7;
            alpha_modifier[2] = (modifiers >> 20) & 0x7;
            if (stage == 0) {
                for (int i = 0; i < 3; ++i) {
                    if (alpha_source[i] == 0xF) {
                        alpha_source[i] = alpha_source[2];
                    }
                }
            }
            float alphas[3];
            for (int i = 0; i < 3; ++i) {
                alphas[i] = AlphaModifier(TevSource(alpha_source[i], primary, previous, held, constant_colour,
                                                    texture0, texture1, texture2, texture3),
                                          alpha_modifier[i]);
            }
            alpha_output = byteround(AlphaCombiner(alpha_op, alphas[0], alphas[1], alphas[2]));
        }
        previous = vec4(clamp(color_output * Multiplier(scales & 3), vec3(0.0), vec3(1.0)),
                        clamp(alpha_output * Multiplier((scales >> 16) & 3), 0.0, 1.0));
        held = next_buffer;
        if (stage < 4) {
            if ((u_update_rgb & (1 << stage)) != 0) {
                next_buffer.rgb = previous.rgb;
            }
            if ((u_update_alpha & (1 << stage)) != 0) {
                next_buffer.a = previous.a;
            }
        }
    }

    if (u_alpha_test != 0 && AlphaTestFails(u_alpha_func, int(previous.a * 255.0))) {
        discard;
    }
    // The fog (FogMode::Fog): the table the game filled turns the fragment's depth into a factor; the colour is mixed
    // towards the fog colour by it (gpu/fragment/fog/fog.cpp's WriteFog).
    if (u_fog_mode == 5) {
        // The PICA's depth: its clip space has z in [-w, 0] (the near clip plane is z <= 0), so GL's window depth
        // 0.5 + 0.5 * z_ndc is 0.5 down to 0 at the near plane and 1 at the far one: -z/w = 2 * depth - 1.
        float fog_depth = clamp(gl_FragCoord.z * 2.0 - 1.0, 0.0, 1.0);
        float fog_index = (u_fog_flip != 0 ? (1.0 - fog_depth) : fog_depth) * 128.0;
        float fog_i = clamp(floor(fog_index), 0.0, 127.0);
        float fog_f = fog_index - fog_i;
        vec2 entry = texelFetch(u_fog_lut, ivec2(int(fog_i), 0), 0).rg;
        float fog_factor = clamp(entry.r + entry.g * fog_f, 0.0, 1.0);
        previous.rgb = mix(u_fog_colour, previous.rgb, fog_factor);
    }
    o_colour = previous;
}
)";

// The composite of a logic op: a full-screen triangle (no attributes: gl_VertexID places it) that reads the draw's source
// texture and a copy of the target's pixels, applies the PICA's bit-by-bit op, and writes the result. Only the pixels the
// draw covered are touched (the stencil buffer carries that, see Surface).
constexpr const char* kCompositeVertexSource = R"(#version 310 es
void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
)";

// Azahar's LogicOp on the 8-bit values (gpu/fbops/output_stage.glsl's mh4n_logic_op): Clear 0, And 1, AndReverse 2,
// Copy 3, Set 4, CopyInverted 5, NoOp 6, Invert 7, Nand 8, Or 9, Nor 10, Xor 11, Equiv 12, AndInverted 13, OrReverse 14,
// OrInverted 15.
constexpr const char* kCompositeFragmentSource = R"(#version 310 es
precision highp float;
uniform sampler2D u_source;
uniform sampler2D u_destination;
uniform int u_logic_op;
out vec4 o_colour;
vec4 LogicOp(int op, vec4 src, vec4 dst) {
    uvec4 s = uvec4(clamp(floor(src * 255.0 + 0.5), 0.0, 255.0));
    uvec4 d = uvec4(clamp(floor(dst * 255.0 + 0.5), 0.0, 255.0));
    uvec4 r = s;
    if (op == 0) {
        r = uvec4(0u);
    } else if (op == 1) {
        r = s & d;
    } else if (op == 2) {
        r = s & ~d;
    } else if (op == 3) {
        r = s;
    } else if (op == 4) {
        r = uvec4(255u);
    } else if (op == 5) {
        r = ~s;
    } else if (op == 7) {
        r = ~d;
    } else if (op == 8) {
        r = ~(s & d);
    } else if (op == 9) {
        r = s | d;
    } else if (op == 10) {
        r = ~(s | d);
    } else if (op == 11) {
        r = s ^ d;
    } else if (op == 12) {
        r = ~(s ^ d);
    } else if (op == 13) {
        r = ~s & d;
    } else if (op == 14) {
        r = s | ~d;
    } else if (op == 15) {
        r = ~s | d;
    }
    return vec4(r & 255u) / 255.0;
}
void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    o_colour = LogicOp(u_logic_op, texelFetch(u_source, at, 0), texelFetch(u_destination, at, 0));
}
)";

struct Composite {
    GLuint program = 0;
    GLuint vao = 0;
    GLint source = -1, destination = -1, op = -1;
};

struct Uniforms {
    GLint source = -1, modifier = -1, ops = -1, scale = -1, constant = -1;
    GLint combiner_colour = -1, update_rgb = -1, update_alpha = -1;
    GLint alpha_test = -1, alpha_func = -1, alpha_ref = -1, scissor_mode = -1, scissor = -1;
    GLint flip_y = -1;
    GLint texture[3] = {-1, -1, -1}, has_texture[3] = {-1, -1, -1}, coordinate[3] = {-1, -1, -1};
    GLint fog_lut = -1, fog_mode = -1, fog_flip = -1, fog_colour = -1;
    GLint proctex = -1, proctex_enable = -1, proctex_config = -1, proctex_lut = -1, proctex_offset = -1,
          proctex_coordinate = -1, proctex_amplitude = -1, proctex_phase = -1, proctex_frequency = -1;
};

// One colour buffer's target: a framebuffer with the colour as a texture (so it can be read back) and a depth
// renderbuffer, kept for as long as the game draws into that buffer. `live` is set at the pass's first draw and cleared
// by EndFrame, so the picture is loaded from guest memory (and the depth cleared) once a frame, not at every draw.
struct Surface {
    GLuint fbo = 0;
    GLuint colour = 0;
    GLuint depth = 0;
    // A logic op's draw goes into `source` (its own framebuffer, sharing the depth buffer, so the depth test and write
    // happen once); the composite reads `source` and `destination` (a copy of the target's pixels) and writes the target.
    GLuint source = 0;
    GLuint source_fbo = 0;
    GLuint destination = 0;
    bool has_stencil = false;    // the depth buffer carries a stencil, where the composite finds the covered pixels
    bool source_ok = false;      // the source framebuffer is complete
    bool allocated = false;      // the attachments have been made for this target's size
    Target target;
    bool live = false;
    bool dirty = false;  // a draw landed in it since the last write-back into guest memory
    u64 used = 0;
};

// A translated PICA vertex stage, compiled for this renderer (DrawVertexProgram), and the key it was made for.
struct VertexProgram {
    shader::VertexShaderConfig config;
    GLuint program = 0;
    GLuint pica_block = GL_INVALID_INDEX;   // the index of its `vs_pica_data` block (bound to point 0)
    GLuint data_block = GL_INVALID_INDEX;   // ... and of its `vs_data` block (point 1)
    Uniforms uniforms;                      // the fragment stage's locations in this program (the same names, new ones)
    unsigned used_inputs = 0;               // bit r: the stage reads input register r
    u64 used = 0;                           // when it was last drawn with (the cache's least recently used goes)
};

// Past this many stages the least recently used program goes: a game has a handful of vertex programs, and a new one
// costs the driver's compile on the draw that needs it.
constexpr std::size_t kMaxPrograms = 48;

struct Objects {
    bool built = false;
    bool tested = false;  // the renderer's own test triangle has been drawn (SelfTest)
    Composite composite;
    GLuint program = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint fog_lut = 0;  // the game's fog table, as a 128x1 RG32F texture (made on first use)
    u32 fog_lut_key = 0;
    GLuint proctex = 0;  // the procedural texture's tables, as a 256x5 RGBA32F texture
    u32 proctex_key = 0;
    Surface* current = nullptr;  // the target of the last Begin
    u64 clock = 0;
    Uniforms uniforms;
    std::unordered_map<u64, Surface> surfaces;  // a colour buffer (address, size) -> its target
    std::unordered_map<u64, GLuint> images;     // a texture's key -> the GL image of its pixels
    std::vector<VertexProgram> programs;        // the draws' own vertex stages (DrawVertexProgram)
    GLuint program_vao = 0;                     // ... their vertex array, vertices and the indices of a draw
    GLuint program_vbo = 0;
    GLuint program_ibo = 0;
    GLuint program_ubo = 0;                     // ... and the two uniform blocks of one draw
    GLuint program_data_ubo = 0;
    std::vector<u8> flipped;                    // the seed, row 0 last (GL's bottom row first)
    std::vector<u8> upload;                     // an image's pixels, row 0 last (the same flip)
};

Objects g_;

std::string ShaderLog(GLuint shader) {
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return std::string();
    }
    std::string text(static_cast<std::size_t>(length), '\0');
    GLsizei written = 0;
    glGetShaderInfoLog(shader, length, &written, text.data());
    text.resize(static_cast<std::size_t>(written));
    return text;
}

bool Compile(GLenum type, const char* source, GLuint* out, std::string* error) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok != GL_TRUE) {
        *error = std::string(type == GL_VERTEX_SHADER ? "the vertex stage" : "the fragment stage") +
                 " did not compile: " + ShaderLog(shader);
        glDeleteShader(shader);
        return false;
    }
    *out = shader;
    return true;
}

// Every uniform the combiner's fragment stage reads, by the name it declares them (the locations differ from one
// program to another: the draws' own vertex stages are linked with the same fragment stage, at their own locations).
void QueryUniforms(GLuint program, Uniforms* u) {
    u->source = glGetUniformLocation(program, "u_source[0]");
    u->modifier = glGetUniformLocation(program, "u_modifier[0]");
    u->ops = glGetUniformLocation(program, "u_ops[0]");
    u->scale = glGetUniformLocation(program, "u_scale[0]");
    u->constant = glGetUniformLocation(program, "u_constant[0]");
    u->combiner_colour = glGetUniformLocation(program, "u_combiner_colour");
    u->update_rgb = glGetUniformLocation(program, "u_update_rgb");
    u->update_alpha = glGetUniformLocation(program, "u_update_alpha");
    u->alpha_test = glGetUniformLocation(program, "u_alpha_test");
    u->alpha_func = glGetUniformLocation(program, "u_alpha_func");
    u->alpha_ref = glGetUniformLocation(program, "u_alpha_ref");
    u->scissor_mode = glGetUniformLocation(program, "u_scissor_mode");
    u->scissor = glGetUniformLocation(program, "u_scissor");
    u->flip_y = glGetUniformLocation(program, "u_flip_y");
    u->fog_lut = glGetUniformLocation(program, "u_fog_lut");
    u->fog_mode = glGetUniformLocation(program, "u_fog_mode");
    u->fog_flip = glGetUniformLocation(program, "u_fog_flip");
    u->fog_colour = glGetUniformLocation(program, "u_fog_colour");
    u->proctex = glGetUniformLocation(program, "u_proctex");
    u->proctex_enable = glGetUniformLocation(program, "u_proctex_enable");
    u->proctex_config = glGetUniformLocation(program, "u_proctex_config");
    u->proctex_lut = glGetUniformLocation(program, "u_proctex_lut");
    u->proctex_offset = glGetUniformLocation(program, "u_proctex_offset");
    u->proctex_coordinate = glGetUniformLocation(program, "u_proctex_coordinate");
    u->proctex_amplitude = glGetUniformLocation(program, "u_proctex_amplitude");
    u->proctex_phase = glGetUniformLocation(program, "u_proctex_phase");
    u->proctex_frequency = glGetUniformLocation(program, "u_proctex_frequency");
    for (int unit = 0; unit < 3; ++unit) {
        const std::string texture = "u_texture" + std::to_string(unit);
        const std::string has = "u_has_texture" + std::to_string(unit);
        const std::string coordinate = "u_coordinate" + std::to_string(unit);
        u->texture[unit] = glGetUniformLocation(program, texture.c_str());
        u->has_texture[unit] = glGetUniformLocation(program, has.c_str());
        u->coordinate[unit] = glGetUniformLocation(program, coordinate.c_str());
    }
}

bool Build(std::string* error) {
    GLuint vertex = 0, fragment = 0;
    if (!Compile(GL_VERTEX_SHADER, kVertexSource, &vertex, error)) {
        return false;
    }
    if (!Compile(GL_FRAGMENT_SHADER, kFragmentSource, &fragment, error)) {
        glDeleteShader(vertex);
        return false;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string text(length > 1 ? static_cast<std::size_t>(length) : 0, '\0');
        if (length > 1) {
            glGetProgramInfoLog(program, length, nullptr, text.data());
        }
        *error = "the program did not link: " + text;
        glDeleteProgram(program);
        return false;
    }
    g_.program = program;
    QueryUniforms(program, &g_.uniforms);
    // The logic op's composite program (a full-screen triangle, no attributes and an empty vertex array).
    GLuint composite_vertex = 0, composite_fragment = 0;
    if (!Compile(GL_VERTEX_SHADER, kCompositeVertexSource, &composite_vertex, error)) {
        return false;
    }
    if (!Compile(GL_FRAGMENT_SHADER, kCompositeFragmentSource, &composite_fragment, error)) {
        glDeleteShader(composite_vertex);
        return false;
    }
    const GLuint composite = glCreateProgram();
    glAttachShader(composite, composite_vertex);
    glAttachShader(composite, composite_fragment);
    glLinkProgram(composite);
    glDeleteShader(composite_vertex);
    glDeleteShader(composite_fragment);
    ok = GL_FALSE;
    glGetProgramiv(composite, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        *error = "the logic op's program did not link";
        glDeleteProgram(composite);
        return false;
    }
    g_.composite.program = composite;
    g_.composite.source = glGetUniformLocation(composite, "u_source");
    g_.composite.destination = glGetUniformLocation(composite, "u_destination");
    g_.composite.op = glGetUniformLocation(composite, "u_logic_op");
    glGenVertexArrays(1, &g_.composite.vao);
    glGenVertexArrays(1, &g_.vao);
    glGenBuffers(1, &g_.vbo);
    glBindVertexArray(g_.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_.vbo);
    // The layout is fixed (kVertexStride and the offsets of HardwareVertex); the locations the stage does not read are
    // described too, so the array is complete for a stage that comes to read them (5, 6 and 7).
    const GLint sizes[kAttribCount] = {4, 4, 2, 2, 2, 1, 4, 3};
    const GLenum types[kAttribCount] = {GL_FLOAT, GL_FLOAT, GL_FLOAT, GL_FLOAT, GL_FLOAT,
                                        GL_FLOAT, GL_FLOAT, GL_FLOAT};
    const unsigned offsets[kAttribCount] = {
        static_cast<unsigned>(offsetof(cpuvertex::HardwareVertex, position)),
        static_cast<unsigned>(offsetof(cpuvertex::HardwareVertex, color)),
        static_cast<unsigned>(offsetof(cpuvertex::HardwareVertex, tex_coord0)),
        static_cast<unsigned>(offsetof(cpuvertex::HardwareVertex, tex_coord1)),
        static_cast<unsigned>(offsetof(cpuvertex::HardwareVertex, tex_coord2)),
        static_cast<unsigned>(offsetof(cpuvertex::HardwareVertex, tex_coord0_w)),
        static_cast<unsigned>(offsetof(cpuvertex::HardwareVertex, normquat)),
        static_cast<unsigned>(offsetof(cpuvertex::HardwareVertex, view))};
    static_assert(sizeof(cpuvertex::HardwareVertex) == kVertexStride, "the batch's vertex is 88 bytes");
    for (unsigned i = 0; i < kAttribCount; ++i) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, sizes[i], types[i], GL_FALSE, kVertexStride,
                              reinterpret_cast<const void*>(static_cast<std::uintptr_t>(offsets[i])));
    }
    glBindVertexArray(0);
    const GLenum error_gl = glGetError();
    if (error_gl != GL_NO_ERROR) {
        char why[96];
        std::snprintf(why, sizeof(why), "the vertex array failed (0x%04X)", static_cast<unsigned>(error_gl));
        *error = why;
        return false;
    }
    g_.built = true;
    return true;
}

// A colour buffer and its size name one target (two screens of one size are two).
u64 SurfaceKey(const Target& target) {
    return static_cast<u64>(target.address) | (static_cast<u64>(target.width & 0x7FF) << 32) |
           (static_cast<u64>(target.height & 0x3FF) << 43) | (static_cast<u64>(target.format & 7) << 53);
}

void DeleteSurface(Surface& surface) {
    if (surface.fbo != 0) {
        glDeleteFramebuffers(1, &surface.fbo);
    }
    if (surface.source_fbo != 0) {
        glDeleteFramebuffers(1, &surface.source_fbo);
    }
    for (GLuint* texture : {&surface.colour, &surface.source, &surface.destination}) {
        if (*texture != 0) {
            glDeleteTextures(1, texture);
        }
    }
    if (surface.depth != 0) {
        glDeleteRenderbuffers(1, &surface.depth);
    }
    surface = Surface{};
}

// The surface's attachments, made (or made again when its size changed): the colour as a texture (so it can be read
// back) and the depth buffer the 3D draws test against, 24 bits where the driver makes one, 16 where it does not.
bool MakeAttachments(Surface& surface, std::string* error) {
    if (surface.allocated) {
        return true;
    }
    const Target& target = surface.target;
    if (surface.fbo == 0) {
        glGenFramebuffers(1, &surface.fbo);
    }
    if (surface.colour == 0) {
        glGenTextures(1, &surface.colour);
    }
    if (surface.depth == 0) {
        glGenRenderbuffers(1, &surface.depth);
    }
    glBindTexture(GL_TEXTURE_2D, surface.colour);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(target.width), static_cast<GLsizei>(target.height), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, surface.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, surface.colour, 0);
    // The depth buffer the 3D draws test against: with a stencil where the driver makes one (the logic op's composite
    // uses it to know the pixels the draw covered), else depth alone. 24 bits where offered, 16 where not.
    const GLenum depth_formats[] = {GL_DEPTH24_STENCIL8, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT16};
    GLenum status = GL_FRAMEBUFFER_UNSUPPORTED;
    GLenum attachment = GL_DEPTH_ATTACHMENT;
    for (const GLenum format : depth_formats) {
        const bool stencil = format == GL_DEPTH24_STENCIL8;
        glBindRenderbuffer(GL_RENDERBUFFER, surface.depth);
        glRenderbufferStorage(GL_RENDERBUFFER, format, static_cast<GLsizei>(target.width),
                              static_cast<GLsizei>(target.height));
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, stencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, surface.depth);
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status == GL_FRAMEBUFFER_COMPLETE) {
            surface.has_stencil = stencil;
            attachment = stencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;
            break;
        }
    }
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        char why[96];
        std::snprintf(why, sizeof(why), "the %ux%u target is not complete (0x%04X)", target.width, target.height,
                      static_cast<unsigned>(status));
        *error = why;
        return false;
    }
    // The logic op's two textures, and the framebuffer its draw goes into (same depth buffer, so one depth test and one
    // depth write per draw).
    surface.source_ok = false;
    if (surface.source == 0) {
        glGenTextures(1, &surface.source);
        glGenTextures(1, &surface.destination);
        glGenFramebuffers(1, &surface.source_fbo);
    }
    for (const GLuint texture : {surface.source, surface.destination}) {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(target.width),
                     static_cast<GLsizei>(target.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, surface.source_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, surface.source, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, surface.depth);
    surface.source_ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, surface.fbo);
    surface.allocated = true;
    return true;
}

// The pass's first draw: the target's picture is loaded from guest memory (so the game's memory fills, and everything the
// previous frame drew before its swap, are there), or cleared to opaque black when there is none; the depth starts empty.
bool StartPass(Surface& surface, const TargetContents* contents, std::string* error) {
    const Target& target = surface.target;
    if (!MakeAttachments(surface, error)) {
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, surface.fbo);
    glViewport(0, 0, static_cast<GLsizei>(target.width), static_cast<GLsizei>(target.height));
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    if (contents != nullptr && contents->rgba8 != nullptr && contents->width == target.width &&
        contents->height == target.height) {
        // GL's texture row 0 is the bottom of the picture; the seed's row 0 is the top (the PICA's), as ReadTarget's
        // flip of the read back says.
        const std::size_t row = static_cast<std::size_t>(target.width) * 4;
        g_.flipped.resize(row * target.height);
        for (u32 y = 0; y < target.height; ++y) {
            const u8* from = contents->rgba8 + static_cast<std::size_t>(y) * row;
            std::copy(from, from + row, g_.flipped.data() + static_cast<std::size_t>(target.height - 1 - y) * row);
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindTexture(GL_TEXTURE_2D, surface.colour);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(target.width),
                        static_cast<GLsizei>(target.height), GL_RGBA, GL_UNSIGNED_BYTE, g_.flipped.data());
    } else {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glDepthMask(GL_TRUE);
    glClearDepthf(1.0f);
    glClear(GL_DEPTH_BUFFER_BIT);
    if (surface.has_stencil) {
        glDisable(GL_SCISSOR_TEST);
        glStencilMask(0xFF);
        glClearStencil(0);
        glClear(GL_STENCIL_BUFFER_BIT);
        glDisable(GL_STENCIL_TEST);
    }
    surface.live = true;
    surface.used = ++g_.clock;
    g_.current = &surface;
    return true;
}

// Past this many colour buffers (a frame's swaps and a few screens), the least recently used one that is not in a pass
// goes: the game gets its buffer back with a fresh image (and starts from what memory holds then).
constexpr std::size_t kMaxSurfaces = 8;

void EvictSurfaces() {
    while (g_.surfaces.size() > kMaxSurfaces) {
        Surface* oldest = nullptr;
        for (auto& [key, surface] : g_.surfaces) {
            if (surface.live || &surface == g_.current) {
                continue;
            }
            if (oldest == nullptr || surface.used < oldest->used) {
                oldest = &surface;
            }
        }
        if (oldest == nullptr) {
            return;
        }
        const u64 key = SurfaceKey(oldest->target);
        DeleteSurface(*oldest);
        g_.surfaces.erase(key);
    }
}

// The GL image of a texture's pixels (uploaded on first sight of its key).
GLuint ImageOf(const Texture& texture) {
    const auto it = g_.images.find(texture.key);
    if (it != g_.images.end()) {
        return it->second;
    }
    if (g_.images.size() >= kMaxImages) {
        for (const auto& [key, image] : g_.images) {
            GLuint gone = image;
            glDeleteTextures(1, &gone);
        }
        g_.images.clear();
    }
    GLuint image = 0;
    glGenTextures(1, &image);
    glBindTexture(GL_TEXTURE_2D, image);
    // The decoder's row 0 is the texture's v = 0 (the PICA's first row of tiles); GL's texture row 0 is v = 0 at the
    // bottom, so the rows are reversed here, as the framebuffer's seed is.
    const std::size_t row = static_cast<std::size_t>(texture.width) * 4;
    const u8* pixels = texture.rgba8;
    if (texture.height > 1) {
        g_.upload.resize(row * texture.height);
        for (u32 y = 0; y < texture.height; ++y) {
            const u8* from = texture.rgba8 + static_cast<std::size_t>(y) * row;
            std::copy(from, from + row, g_.upload.data() + static_cast<std::size_t>(texture.height - 1 - y) * row);
        }
        pixels = g_.upload.data();
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(texture.width), static_cast<GLsizei>(texture.height),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    g_.images.emplace(texture.key, image);
    return image;
}

// The PICA's wrap modes: 0 clamp, 1 clamp to border (no border colour here: clamped), 2 and 6 repeat, 3 and 7
// mirrored repeat, 4 and 5 clamp (the rectangle modes).
GLint WrapMode(u32 mode) {
    switch (mode & 7) {
    case 2:
    case 6: return GL_REPEAT;
    case 3:
    case 7: return GL_MIRRORED_REPEAT;
    default: return GL_CLAMP_TO_EDGE;
    }
}

GLenum BlendEquation(u32 equation) {
    switch (equation) {
    case 1: return GL_FUNC_SUBTRACT;
    case 2: return GL_FUNC_REVERSE_SUBTRACT;
    case 3: return GL_MIN;
    case 4: return GL_MAX;
    default: return GL_FUNC_ADD;
    }
}

GLenum BlendFactor(u32 factor) {
    switch (factor) {
    case 0: return GL_ZERO;
    case 2: return GL_SRC_COLOR;
    case 3: return GL_ONE_MINUS_SRC_COLOR;
    case 4: return GL_DST_COLOR;
    case 5: return GL_ONE_MINUS_DST_COLOR;
    case 6: return GL_SRC_ALPHA;
    case 7: return GL_ONE_MINUS_SRC_ALPHA;
    case 8: return GL_DST_ALPHA;
    case 9: return GL_ONE_MINUS_DST_ALPHA;
    case 10: return GL_CONSTANT_COLOR;
    case 11: return GL_ONE_MINUS_CONSTANT_COLOR;
    case 12: return GL_CONSTANT_ALPHA;
    case 13: return GL_ONE_MINUS_CONSTANT_ALPHA;
    case 14: return GL_SRC_ALPHA_SATURATE;
    default: return GL_ONE;
    }
}

GLenum CompareFunc(u32 function) {
    switch (function) {
    case 0: return GL_NEVER;
    case 2: return GL_EQUAL;
    case 3: return GL_NOTEQUAL;
    case 4: return GL_LESS;
    case 5: return GL_LEQUAL;
    case 6: return GL_GREATER;
    case 7: return GL_GEQUAL;
    default: return GL_ALWAYS;
    }
}

void SetBlend(const DrawState& state) {
    if (!state.blend) {
        glDisable(GL_BLEND);
        return;
    }
    glEnable(GL_BLEND);
    glBlendEquationSeparate(BlendEquation(state.blend_equation_rgb), BlendEquation(state.blend_equation_alpha));
    glBlendFuncSeparate(BlendFactor(state.blend_source_rgb), BlendFactor(state.blend_dest_rgb),
                        BlendFactor(state.blend_source_alpha), BlendFactor(state.blend_dest_alpha));
    glBlendColor(state.blend_constant[0], state.blend_constant[1], state.blend_constant[2], state.blend_constant[3]);
}

// A half-precision word as a float: the fog table's entries are two of them (r at the low word, g at the high one).
float HalfToFloat(u32 half) {
    const u32 sign = (half & 0x8000u) << 16;
    const u32 exponent = (half >> 10) & 0x1Fu;
    const u32 mantissa = half & 0x3FFu;
    u32 bits;
    if (exponent == 0) {
        bits = sign | (mantissa << 13);
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// The fog: the mode and its flip and colour as uniforms, the game's 128-entry table as an RG32F texture on unit 3 (the
// entries are uploaded when they changed). A draw with no table has no fog.
void SetFog(const FragmentState& fragment, const Uniforms& u) {
    const bool fog = fragment.fog_lut != nullptr && (fragment.fog_mode & 7) == 5;
    glUniform1i(u.fog_mode, fog ? 5 : 0);
    glUniform1i(u.fog_flip, fragment.fog_flip ? 1 : 0);
    glUniform3f(u.fog_colour, fragment.fog_colour[0], fragment.fog_colour[1], fragment.fog_colour[2]);
    if (!fog) {
        return;
    }
    if (g_.fog_lut == 0) {
        glGenTextures(1, &g_.fog_lut);
        glBindTexture(GL_TEXTURE_2D, g_.fog_lut);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    if (fragment.fog_lut_key != g_.fog_lut_key) {
        float values[128 * 2];
        for (u32 i = 0; i < 128; ++i) {
            values[i * 2 + 0] = HalfToFloat(fragment.fog_lut[i] & 0xFFFFu);
            values[i * 2 + 1] = HalfToFloat(fragment.fog_lut[i] >> 16);
        }
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, g_.fog_lut);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, 128, 1, 0, GL_RG, GL_FLOAT, values);
        g_.fog_lut_key = fragment.fog_lut_key;
    }
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, g_.fog_lut);
    glUniform1i(u.fog_lut, 3);
    glActiveTexture(GL_TEXTURE0);
}

// The procedural texture's five tables as one key (a changed entry gives a new one, so they upload once).
u32 ProcTexKey(const ProcTexState& proctex) {
    u32 hash = 2166136261u;
    const auto mix = [&hash](const std::array<u32, 128>& table) {
        for (const u32 word : table) {
            hash = (hash ^ word) * 16777619u;
        }
    };
    const auto mix_colours = [&hash](const std::array<u32, 256>& table) {
        for (const u32 word : table) {
            hash = (hash ^ word) * 16777619u;
        }
    };
    mix(*proctex.noise_map);
    mix(*proctex.color_map);
    mix(*proctex.alpha_map);
    mix_colours(*proctex.color_table);
    mix_colours(*proctex.color_diff_table);
    return hash != 0 ? hash : 1;
}

// The procedural texture (texture unit 3): its registers as uniforms, its tables as a 256x5 RGBA32F texture on unit 4
// (row 0 noise, 1 colour map, 2 alpha map, 3 the colours, 4 the steps), uploaded when an entry changed. The decoding is
// lavilao/pocket-native's proctex.rs.
void SetProcTex(const ProcTexState& proctex, const Uniforms& u) {
    const bool enable = proctex.enable && proctex.noise_map != nullptr && proctex.color_map != nullptr &&
                        proctex.alpha_map != nullptr && proctex.color_table != nullptr && proctex.color_diff_table != nullptr;
    glUniform1i(u.proctex_enable, enable ? 1 : 0);
    if (!enable) {
        return;
    }
    glUniform1i(u.proctex_config, static_cast<GLint>(proctex.config));
    glUniform1i(u.proctex_lut, static_cast<GLint>(proctex.lut));
    glUniform1i(u.proctex_offset, static_cast<GLint>(proctex.offset));
    glUniform1i(u.proctex_coordinate, static_cast<GLint>(std::min(proctex.coordinate, 2u)));
    glUniform2f(u.proctex_amplitude, static_cast<GLfloat>(static_cast<s16>(proctex.noise[0] & 0xFFFFu)) / 4095.0f,
                static_cast<GLfloat>(static_cast<s16>(proctex.noise[1] & 0xFFFFu)) / 4095.0f);
    glUniform2f(u.proctex_phase, static_cast<GLfloat>(proctex.noise[0] >> 16) / 4096.0f,
                static_cast<GLfloat>(proctex.noise[1] >> 16) / 4096.0f);
    glUniform2f(u.proctex_frequency, HalfToFloat(proctex.frequency & 0xFFFFu), HalfToFloat(proctex.frequency >> 16));
    const u32 key = ProcTexKey(proctex);
    if (g_.proctex == 0) {
        glGenTextures(1, &g_.proctex);
        glBindTexture(GL_TEXTURE_2D, g_.proctex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    if (key != g_.proctex_key) {
        float values[256 * 5 * 4] = {};
        const auto maps = [&values](u32 row, const std::array<u32, 128>& table) {
            for (u32 i = 0; i < 128; ++i) {
                const u32 word = table[i];
                const s32 step = static_cast<s32>(word << 12) >> 20;  // a signed 12-bit step
                values[(row * 256 + i) * 4 + 0] = static_cast<float>(word & 0xFFFu) / 4095.0f;
                values[(row * 256 + i) * 4 + 1] = static_cast<float>(step) / 4095.0f;
            }
        };
        maps(0, *proctex.noise_map);
        maps(1, *proctex.color_map);
        maps(2, *proctex.alpha_map);
        for (u32 i = 0; i < 256; ++i) {
            const u32 word = (*proctex.color_table)[i];
            const u32 difference = (*proctex.color_diff_table)[i];
            for (u32 channel = 0; channel < 4; ++channel) {
                values[(3 * 256 + i) * 4 + channel] = static_cast<float>((word >> (8 * channel)) & 0xFFu);
                // The steps come halved, signed bytes (a step of -128 is -256).
                values[(4 * 256 + i) * 4 + channel] =
                    static_cast<float>(static_cast<s8>((difference >> (8 * channel)) & 0xFFu)) * 2.0f;
            }
        }
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, g_.proctex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 256, 5, 0, GL_RGBA, GL_FLOAT, values);
        g_.proctex_key = key;
    }
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, g_.proctex);
    glUniform1i(u.proctex, 4);
    glActiveTexture(GL_TEXTURE0);
}

void SetCull(const DrawState& state) {
    if (state.cull == 0) {
        glDisable(GL_CULL_FACE);
        return;
    }
    glEnable(GL_CULL_FACE);
    // As Azahar maps them: the keep-one-winding modes cull the back, or the front when the framebuffer is flipped.
    const bool front = (state.cull == 2) != state.flipped;
    glCullFace(state.cull == 3 ? GL_FRONT_AND_BACK : (front ? GL_FRONT : GL_BACK));
    glFrontFace(state.cull == 2 ? GL_CW : GL_CCW);
}

}  // namespace

bool IsPassStarted(const Target& target) {
    const auto it = g_.surfaces.find(SurfaceKey(target));
    return it != g_.surfaces.end() && it->second.live;
}

// The renderer's own test draw (below ApplyUniforms): once the program is built.
void SelfTest();

bool Begin(const Target& target, const TargetContents* contents, std::string* error) {
    if (!MakeCurrent(error)) {
        return false;
    }
    if (target.width == 0 || target.height == 0) {
        *error = "the draw names an empty target";
        return false;
    }
    if (!g_.built && !Build(error)) {
        return false;
    }
    if (!g_.tested) {
        // (a GL context is current by now: the program is built, so the renderer can try itself once)
        g_.tested = true;
        SelfTest();
    }
    // A draw into another colour buffer (the other screen, or the frame after a swap) starts that buffer's pass: its
    // picture is loaded once, at the first draw into it.
    Surface* surface = &g_.surfaces[SurfaceKey(target)];
    surface->target = target;
    surface->used = ++g_.clock;
    if (!surface->live && !StartPass(*surface, contents, error)) {
        return false;
    }
    EvictSurfaces();
    surface = &g_.surfaces[SurfaceKey(target)];
    g_.current = surface;
    glBindFramebuffer(GL_FRAMEBUFFER, surface->fbo);
    glViewport(0, 0, static_cast<GLsizei>(surface->target.width), static_cast<GLsizei>(surface->target.height));
    glDisable(GL_SCISSOR_TEST);
    return true;
}

void EndFrame(u32 address) {
    bool found = false;
    for (auto& [key, surface] : g_.surfaces) {
        if (surface.target.address == address) {
            surface.live = false;
            found = true;
        }
    }
    if (!found) {
        // (the game swapped a buffer nothing was drawn into: every pass is over as far as this frame goes)
        for (auto& [key, surface] : g_.surfaces) {
            surface.live = false;
        }
    }
}

void EndPasses() {
    for (auto& entry : g_.surfaces) {
        entry.second.live = false;
    }
}

// The fixed-function state a draw's registers ask for (gles_render.h's DrawState). The caller has bound the target's
// framebuffer and the program it draws with.
void ApplyFixedState(const DrawState& state) {
    glViewport(static_cast<GLint>(state.viewport_x), static_cast<GLint>(state.viewport_y),
               static_cast<GLsizei>(state.viewport_width), static_cast<GLsizei>(state.viewport_height));
    if (state.scissor_width != 0 && state.scissor_height != 0) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(state.scissor_x, state.scissor_y, static_cast<GLsizei>(state.scissor_width),
                  static_cast<GLsizei>(state.scissor_height));
    } else {
        glDisable(GL_SCISSOR_TEST);
    }
    if (state.depth_test) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    glDepthFunc(CompareFunc(state.depth_func));
    glDepthMask(state.depth_write ? GL_TRUE : GL_FALSE);
    glColorMask((state.colour_write & 1) != 0, (state.colour_write & 2) != 0, (state.colour_write & 4) != 0,
                (state.colour_write & 8) != 0);
    SetBlend(state);
    SetCull(state);

}

// The fragment stage's uniforms, the three texture units and the tables (the caller has made its program current).
void ApplyUniforms(const DrawState& state, const Uniforms& u) {
    GLint sources[6], modifiers[6], ops[6], scales[6];
    GLfloat constants[24];
    for (int stage = 0; stage < 6; ++stage) {
        const CombinerStage& s = state.fragment.stages[stage];
        sources[stage] = static_cast<GLint>(s.sources);
        modifiers[stage] = static_cast<GLint>(s.modifiers);
        ops[stage] = static_cast<GLint>(s.ops);
        scales[stage] = static_cast<GLint>(s.scales);
        for (int channel = 0; channel < 4; ++channel) {
            constants[stage * 4 + channel] = static_cast<GLfloat>((s.constant >> (8 * channel)) & 0xFF) / 255.0f;
        }
    }
    glUniform1iv(u.source, 6, sources);
    glUniform1iv(u.modifier, 6, modifiers);
    glUniform1iv(u.ops, 6, ops);
    glUniform1iv(u.scale, 6, scales);
    glUniform4fv(u.constant, 6, constants);
    const u32 buffer = state.fragment.combiner_buffer_colour;
    glUniform4f(u.combiner_colour, static_cast<GLfloat>(buffer & 0xFF) / 255.0f,
                static_cast<GLfloat>((buffer >> 8) & 0xFF) / 255.0f,
                static_cast<GLfloat>((buffer >> 16) & 0xFF) / 255.0f,
                static_cast<GLfloat>((buffer >> 24) & 0xFF) / 255.0f);
    glUniform1i(u.update_rgb, static_cast<GLint>(state.fragment.update_rgb & 0xF));
    glUniform1i(u.update_alpha, static_cast<GLint>(state.fragment.update_alpha & 0xF));
    glUniform1i(u.alpha_test, state.fragment.alpha_test ? 1 : 0);
    glUniform1i(u.alpha_func, static_cast<GLint>(state.fragment.alpha_func & 7));
    glUniform1i(u.alpha_ref, static_cast<GLint>(state.fragment.alpha_ref & 0xFF));
    glUniform1i(u.scissor_mode, static_cast<GLint>(state.fragment.scissor_mode & 3));
    glUniform4i(u.scissor, state.fragment.scissor_x0, state.fragment.scissor_y0, state.fragment.scissor_x1,
                state.fragment.scissor_y1);
    glUniform1i(u.flip_y, state.flipped ? 1 : 0);

    for (int unit = 0; unit < 3; ++unit) {
        const bool set = unit < state.texture_count && state.textures[unit].rgba8 != nullptr;
        glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
        if (set) {
            const Texture& texture = state.textures[unit];
            glBindTexture(GL_TEXTURE_2D, ImageOf(texture));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, WrapMode(texture.wrap_s));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, WrapMode(texture.wrap_t));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texture.mag_linear ? GL_LINEAR : GL_NEAREST);
            // (the images have one level: a minification filter without mipmaps, whatever the LOD register says)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texture.min_linear ? GL_LINEAR : GL_NEAREST);
            glUniform1i(u.texture[unit], unit);
        } else {
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glUniform1i(u.has_texture[unit], set ? 1 : 0);
        glUniform1i(u.coordinate[unit], static_cast<GLint>(set ? std::min(state.textures[unit].coordinate, 2u) : 0u));
    }
    SetFog(state.fragment, u);
    SetProcTex(state.fragment.proctex, u);
}

// The renderer's own test draw, once per built program: a red triangle in clip space through the batch program into a
// 64x64 framebuffer of its own, read back. It tells the log what a black screen otherwise leaves in doubt — that the
// driver compiled both stages, that a draw lands, and that pixels come back — and names the GL error when they do not.
void SelfTest() {
    GLuint fbo = 0, colour = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &colour);
    glBindTexture(GL_TEXTURE_2D, colour);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colour, 0);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    u8 pixel[4] = {0, 0, 0, 0};
    GLenum error_gl = GL_NO_ERROR;
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        error_gl = status;
    } else {
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glViewport(0, 0, 64, 64);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // The fragment state: every stage replaces with the primary colour (the vertex colour), as a draw with no
        // texture does, and nothing else is on.
        DrawState state;
        // (the whole target: a draw's own viewport comes from its registers, this one has none)
        state.viewport_x = 0.0f;
        state.viewport_y = 0.0f;
        state.viewport_width = 64.0f;
        state.viewport_height = 64.0f;
        state.depth_test = false;
        state.depth_write = false;
        state.colour_write = 0xF;
        state.cull = 0;
        state.blend = false;
        state.logic_op = 3;  // Copy
        state.fragment.alpha_test = false;
        state.fragment.fog_mode = 0;
        for (int stage = 0; stage < 6; ++stage) {
            CombinerStage& s = state.fragment.stages[stage];
            s.sources = 0;  // PrimaryColor (source 0) in all three colour sources and the alpha ones
            s.modifiers = 0;
            s.ops = 0;  // Replace
            s.scales = 0;
        }

        // A triangle over the middle of the target: the PICA's clip space has z in [-w, 0], so z = 0 is the near plane.
        cpuvertex::HardwareVertex triangle[3]{};
        const float positions[3][4] = {{-0.8f, -0.8f, 0.0f, 1.0f}, {0.8f, -0.8f, 0.0f, 1.0f}, {0.0f, 0.8f, 0.0f, 1.0f}};
        for (int i = 0; i < 3; ++i) {
            std::copy(positions[i], positions[i] + 4, triangle[i].position);
            triangle[i].color[0] = 1.0f;  // red
            triangle[i].color[3] = 1.0f;
        }

        glUseProgram(g_.program);
        ApplyFixedState(state);
        ApplyUniforms(state, g_.uniforms);
        glBindVertexArray(g_.vao);
        glBindBuffer(GL_ARRAY_BUFFER, g_.vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(triangle)), triangle, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
        error_gl = glGetError();
        glUseProgram(0);
        glBindVertexArray(0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &colour);
    if (error_gl != GL_NO_ERROR) {
        MH4N_LOG(Error, kLog, "the renderer's own test triangle failed (0x%04X): its draws will not be seen",
                 static_cast<unsigned>(error_gl));
    } else {
        MH4N_LOG(Info, kLog, "the renderer's own test triangle: the centre pixel came out r {}, g {}, b {}, a {}",
                 static_cast<unsigned>(pixel[0]), static_cast<unsigned>(pixel[1]), static_cast<unsigned>(pixel[2]),
                 static_cast<unsigned>(pixel[3]));
    }
}

// Issues a draw into the current surface, with the logic op emulated where GLES has none: `issue` draws the geometry,
// its vertices already bound.
bool Submit(const DrawState& state, const std::function<void()>& issue, std::string* error) {
    Surface& surface = *g_.current;
    const GLsizei w = static_cast<GLsizei>(surface.target.width);
    const GLsizei h = static_cast<GLsizei>(surface.target.height);
    const bool logic = state.logic_op != 3 && state.logic_op != 6;
    const bool emulate = logic && surface.source_ok && surface.has_stencil && g_.composite.program != 0;
    if (emulate) {
        // The draw goes into the source texture; the depth test writes the depth buffer the two framebuffers share, and
        // the stencil marks the pixels that passed, so the composite leaves the others as they were.
        glBindFramebuffer(GL_FRAMEBUFFER, surface.source_fbo);
        glDisable(GL_SCISSOR_TEST);
        glStencilMask(0xFF);
        glClearStencil(0);
        glClear(GL_STENCIL_BUFFER_BIT);
        if (state.scissor_width != 0 && state.scissor_height != 0) {
            glEnable(GL_SCISSOR_TEST);
            glScissor(state.scissor_x, state.scissor_y, static_cast<GLsizei>(state.scissor_width),
                      static_cast<GLsizei>(state.scissor_height));
        }
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        issue();
        glDisable(GL_STENCIL_TEST);
        // Then the composite: a copy of the target's pixels, the source, and the op, over the marked pixels only.
        glBindFramebuffer(GL_FRAMEBUFFER, surface.fbo);
        glBindTexture(GL_TEXTURE_2D, surface.destination);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
        glViewport(0, 0, w, h);
        glUseProgram(g_.composite.program);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, surface.source);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, surface.destination);
        glUniform1i(g_.composite.source, 0);
        glUniform1i(g_.composite.destination, 1);
        glUniform1i(g_.composite.op, static_cast<GLint>(state.logic_op));
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glColorMask((state.colour_write & 1) != 0, (state.colour_write & 2) != 0, (state.colour_write & 4) != 0,
                    (state.colour_write & 8) != 0);
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_EQUAL, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glBindVertexArray(g_.composite.vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_STENCIL_TEST);
        glActiveTexture(GL_TEXTURE0);
    } else {
        if (state.logic_op == 6) {
            // NoOp writes no colour; the draw's depth write stays (as the Vulkan path's colour write mask of 0 keeps it).
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        }
        issue();
    }
    const GLenum error_gl = glGetError();
    if (error_gl != GL_NO_ERROR) {
        char why[96];
        std::snprintf(why, sizeof(why), "the draw failed (0x%04X)", static_cast<unsigned>(error_gl));
        *error = why;
        return false;
    }
    g_.current->dirty = true;  // the picture its guest memory holds is out of date until the next write-back
    return true;
}

// ---- the draws' own vertex stages (DrawVertexProgram) ---------------------------------------------------------

namespace {

// A vertex format's GL type, lane count and whether the stage reads it as an integer (vertex_loader.h's conversion).
struct GlVertexFormat {
    GLenum type = GL_FLOAT;
    GLint size = 4;
    bool integer = false;
};

GlVertexFormat GlFormatOf(vertex::VertexFormat format) {
    switch (format) {
    case vertex::VertexFormat::R8_SINT:
        return {GL_BYTE, 1, true};
    case vertex::VertexFormat::R8G8_SINT:
        return {GL_BYTE, 2, true};
    case vertex::VertexFormat::R8G8B8A8_SINT:
        return {GL_BYTE, 4, true};
    case vertex::VertexFormat::R8_UINT:
        return {GL_UNSIGNED_BYTE, 1, true};
    case vertex::VertexFormat::R8G8_UINT:
        return {GL_UNSIGNED_BYTE, 2, true};
    case vertex::VertexFormat::R8G8B8A8_UINT:
        return {GL_UNSIGNED_BYTE, 4, true};
    case vertex::VertexFormat::R16_SINT:
        return {GL_SHORT, 1, true};
    case vertex::VertexFormat::R16G16_SINT:
        return {GL_SHORT, 2, true};
    case vertex::VertexFormat::R16G16B16A16_SINT:
        return {GL_SHORT, 4, true};
    case vertex::VertexFormat::R32_FLOAT:
        return {GL_FLOAT, 1, false};
    case vertex::VertexFormat::R32G32_FLOAT:
        return {GL_FLOAT, 2, false};
    case vertex::VertexFormat::R32G32B32_FLOAT:
        return {GL_FLOAT, 3, false};
    case vertex::VertexFormat::R32G32B32A32_FLOAT:
        break;
    }
    return {GL_FLOAT, 4, false};
}

// The stage for `config`, translated and compiled (a miss pays the driver's compile: the draw that needs it).
bool MakeVertexProgram(const pica::PicaState& state, const shader::VertexShaderConfig& config, VertexProgram* out,
                       std::string* error) {
    shader::TranslateResult translated = shader::GenerateVertexGlsl(state.vs, config);
    if (!translated.ok()) {
        *error = translated.error;
        return false;
    }
    std::string es;
    if (!ToEsVertexStage(translated.shader->glsl, &es, error)) {
        return false;
    }
    GLuint vertex = 0, fragment = 0;
    if (!Compile(GL_VERTEX_SHADER, es.c_str(), &vertex, error)) {
        return false;
    }
    if (!Compile(GL_FRAGMENT_SHADER, kFragmentSource, &fragment, error)) {
        glDeleteShader(vertex);
        return false;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    // The fragment program's varyings are the stage's, by name (gles_shader.h), so the two link as they are.
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string text(length > 1 ? static_cast<std::size_t>(length) : 0, '\0');
        if (length > 1) {
            glGetProgramInfoLog(program, length, nullptr, text.data());
        }
        *error = "the translated vertex stage did not link with the fragment stage: " + text;
        glDeleteProgram(program);
        return false;
    }
    const GLuint pica_block = glGetUniformBlockIndex(program, "vs_pica_data");
    const GLuint data_block = glGetUniformBlockIndex(program, "vs_data");
    if (pica_block == GL_INVALID_INDEX || data_block == GL_INVALID_INDEX) {
        *error = "the translated vertex stage has no uniform blocks to bind";
        glDeleteProgram(program);
        return false;
    }
    QueryUniforms(program, &out->uniforms);
    glUniformBlockBinding(program, pica_block, 0);
    glUniformBlockBinding(program, data_block, 1);
    out->program = program;
    out->pica_block = pica_block;
    out->data_block = data_block;
    out->used_inputs = translated.shader->used_inputs;
    return true;
}

// The draw's indices with its base vertex added (GLES 3.1 has no base-vertex draw; vertex_loader.h gives the list
// relative to buffer vertex 0).
void MakeIndices(const vertex::LoadedDraw& loaded, std::vector<u8>& out, GLenum* type) {
    const vertex::IndexType kind = loaded.index_type;
    const std::size_t size = kind == vertex::IndexType::U8 ? 1 : kind == vertex::IndexType::U16 ? 2 : 4;
    *type = kind == vertex::IndexType::U8 ? GL_UNSIGNED_BYTE
            : kind == vertex::IndexType::U16 ? GL_UNSIGNED_SHORT
                                             : GL_UNSIGNED_INT;
    out.resize(static_cast<std::size_t>(loaded.count) * size);
    for (u32 i = 0; i < loaded.count; ++i) {
        u32 value = 0;
        std::memcpy(&value, loaded.index_data.data() + static_cast<std::size_t>(i) * size, size);
        // (the loader's base vertex is signed: an index below the first vertex of the draw stays at the first)
        s32 moved = static_cast<s32>(value) + loaded.base_vertex;
        moved = std::max(0, moved);
        if (size == 1) {
            out[i] = static_cast<u8>(std::min(moved, 255));
        } else if (size == 2) {
            const u16 entry = static_cast<u16>(std::min(moved, 65535));
            std::memcpy(&out[static_cast<std::size_t>(i) * 2], &entry, 2);
        } else {
            const u32 entry = static_cast<u32>(moved);
            std::memcpy(&out[static_cast<std::size_t>(i) * 4], &entry, 4);
        }
    }
}

}  // namespace

bool DrawVertexProgram(const VertexDraw& draw, const DrawState& state, std::string* error) {
    if (!g_.built || g_.current == nullptr) {
        *error = "the rasterizer has no target (Begin makes it)";
        return false;
    }
    if (draw.state == nullptr || draw.loaded == nullptr) {
        *error = "the draw has no vertex stage of its own";
        return false;
    }
    const vertex::LoadedDraw& loaded = *draw.loaded;
    const pica::PicaState& pica_state = *draw.state;
    if (loaded.count == 0) {
        return true;
    }
    if (loaded.topology == vertex::Topology::Shader) {
        *error = "a shader topology's primitives come from the geometry stage, which this renderer has not";
        return false;
    }
    if (loaded.buffers.empty()) {
        *error = "the draw has no vertex buffer to read";
        return false;
    }

    // The stage: the draw's own PICA program, without the clip distances GLSL ES has not (gles_shader.h).
    shader::ConfigOptions options;
    options.use_clip_planes = false;
    options.geometry_stage = false;
    options.capture_outputs = false;
    options.exact_arithmetic = true;
    const shader::VertexShaderConfig config = shader::MakeVertexShaderConfig(pica_state, &loaded, options);
    VertexProgram* stage = nullptr;
    for (VertexProgram& candidate : g_.programs) {
        if (candidate.config == config) {
            stage = &candidate;
            break;
        }
    }
    if (stage == nullptr) {
        VertexProgram made;
        made.config = config;
        if (!MakeVertexProgram(pica_state, config, &made, error)) {
            return false;  // (the caller takes the CPU path: a program this renderer cannot run is never fatal)
        }
        if (g_.programs.size() >= kMaxPrograms) {
            auto oldest = std::min_element(g_.programs.begin(), g_.programs.end(),
                                           [](const VertexProgram& a, const VertexProgram& b) { return a.used < b.used; });
            glDeleteProgram(oldest->program);
            *oldest = made;
            stage = &*oldest;
        } else {
            g_.programs.push_back(made);
            stage = &g_.programs.back();
        }
    }
    stage->used = ++g_.clock;

    // The two uniform blocks of the draw (vertex_shader.h: PicaUniformData, then VertexUniformData).
    struct Blocks {
        shader::PicaUniformData pica;
        shader::VertexUniformData data;
    };
    static thread_local Blocks blocks;
    shader::WritePicaUniformData(pica_state.vs, &blocks.pica);
    blocks.data = shader::MakeVertexUniformData(pica_state, state.flipped);
    if (g_.program_ubo == 0) {
        glGenBuffers(1, &g_.program_ubo);
        glGenBuffers(1, &g_.program_data_ubo);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, g_.program_ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(blocks.pica), &blocks.pica, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, g_.program_data_ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(blocks.data), &blocks.data, GL_STREAM_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, g_.program_ubo);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, g_.program_data_ubo);

    // The vertices: the loader's bindings one after another in one buffer, each attribute where its binding puts it
    // (vertex_loader.h: the strides are the loader's, the offsets the attributes').
    static thread_local std::vector<u8> vertices;
    static thread_local std::vector<u32> starts;
    vertices.clear();
    starts.assign(loaded.buffers.size(), 0);
    for (std::size_t b = 0; b < loaded.buffers.size(); ++b) {
        const vertex::VertexBuffer& buffer = loaded.buffers[b];
        starts[b] = static_cast<u32>(vertices.size());
        vertices.insert(vertices.end(), buffer.data.begin(), buffer.data.end());
    }
    if (g_.program_vao == 0) {
        glGenVertexArrays(1, &g_.program_vao);
        glGenBuffers(1, &g_.program_vbo);
        glGenBuffers(1, &g_.program_ibo);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, g_.current->fbo);
    glUseProgram(stage->program);
    ApplyFixedState(state);
    ApplyUniforms(state, stage->uniforms);
    glBindVertexArray(g_.program_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_.program_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size()), vertices.data(), GL_STREAM_DRAW);
    for (u32 r = 0; r < 16; ++r) {
        const bool used = (stage->used_inputs & (1u << r)) != 0;
        const vertex::VertexAttribute& attribute = loaded.attributes[r];
        if (!used || attribute.binding >= loaded.buffers.size()) {
            glDisableVertexAttribArray(r);
            continue;
        }
        const vertex::VertexBuffer& buffer = loaded.buffers[attribute.binding];
        const GlVertexFormat format = GlFormatOf(attribute.format);
        const void* offset = reinterpret_cast<const void*>(
            static_cast<std::uintptr_t>(starts[attribute.binding] + attribute.offset));
        glVertexAttribDivisor(r, buffer.per_instance ? 1 : 0);
        if (format.integer) {
            glVertexAttribIPointer(r, format.size, format.type, static_cast<GLsizei>(buffer.stride), offset);
        } else {
            glVertexAttribPointer(r, format.size, format.type, GL_FALSE, static_cast<GLsizei>(buffer.stride), offset);
        }
        glEnableVertexAttribArray(r);
    }

    const GLenum mode = loaded.topology == vertex::Topology::TriangleStrip
                            ? GL_TRIANGLE_STRIP
                            : loaded.topology == vertex::Topology::TriangleFan ? GL_TRIANGLE_FAN : GL_TRIANGLES;
    if (loaded.index_type == vertex::IndexType::None) {
        const GLsizei count = static_cast<GLsizei>(loaded.count);
        return Submit(state, [mode, count] { glDrawArrays(mode, 0, count); }, error);
    }
    static thread_local std::vector<u8> indices;
    GLenum index_type = GL_UNSIGNED_SHORT;
    MakeIndices(loaded, indices, &index_type);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_.program_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size()), indices.data(), GL_STREAM_DRAW);
    const GLsizei count = static_cast<GLsizei>(loaded.count);
    return Submit(
        state,
        [mode, index_type, count] {
            glDrawElements(mode, count, index_type, nullptr);
        },
        error);
}

bool DrawTriangles(const void* vertices, u32 count, const DrawState& state, std::string* error) {
    if (count == 0) {
        return true;
    }
    if (!g_.built || g_.current == nullptr) {
        *error = "the rasterizer has no target (Begin makes it)";
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, g_.current->fbo);
    glUseProgram(g_.program);
    ApplyFixedState(state);
    ApplyUniforms(state, g_.uniforms);
    glBindVertexArray(g_.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(static_cast<std::size_t>(count) * kVertexStride), vertices,
                 GL_STREAM_DRAW);
    return Submit(state, [count] { glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(count)); }, error);
}

namespace {

// A surface's colour buffer as RGBA8 rows, row 0 the PICA's first row of the picture (ReadTarget's flip).
bool ReadSurface(Surface& surface, std::vector<u8>& out, u32* width, u32* height) {
    const u32 w = surface.target.width, h = surface.target.height;
    out.resize(static_cast<std::size_t>(w) * h * 4);
    glBindFramebuffer(GL_FRAMEBUFFER, surface.fbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h), GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    if (glGetError() != GL_NO_ERROR) {
        return false;
    }
    // GL's row 0 is the bottom of the picture; the PICA's is the top, and so is what is written to guest memory.
    std::vector<u8> row(static_cast<std::size_t>(w) * 4);
    for (u32 y = 0; y < h / 2; ++y) {
        u8* top = out.data() + static_cast<std::size_t>(y) * w * 4;
        u8* bottom = out.data() + static_cast<std::size_t>(h - 1 - y) * w * 4;
        std::copy(top, top + row.size(), row.begin());
        std::copy(bottom, bottom + row.size(), top);
        std::copy(row.begin(), row.end(), bottom);
    }
    *width = w;
    *height = h;
    return true;
}

}  // namespace

bool WriteBackDrawn(const StorePicture& store) {
    if (!g_.built) {
        return true;
    }
    bool ok = true;
    for (auto& entry : g_.surfaces) {
        Surface& surface = entry.second;
        if (!surface.dirty) {
            continue;
        }
        surface.dirty = false;
        std::vector<u8> pixels;
        u32 w = 0, h = 0;
        if (!ReadSurface(surface, pixels, &w, &h)) {
            MH4N_LOG(Error, kLog, "a {}x{} colour buffer could not be read back for guest memory",
                     surface.target.width, surface.target.height);
            ok = false;
            continue;
        }
        if (!store(surface.target.address, w, h, surface.target.format, pixels.data())) {
            ok = false;
        }
    }
    return ok;
}

bool ReadTarget(u32 address, std::vector<u8>& out, u32* width, u32* height, std::string* error) {
    if (!g_.built) {
        *error = "there is no target to read";
        return false;
    }
    // The buffer the game swapped, when a draw went into it; otherwise the last one drawn.
    Surface* surface = nullptr;
    for (auto& [key, candidate] : g_.surfaces) {
        if (candidate.target.address != address) {
            continue;
        }
        if (surface == nullptr || candidate.live) {
            surface = &candidate;
        }
    }
    if (surface == nullptr) {
        surface = g_.current;
    }
    if (surface == nullptr) {
        *error = "there is no target to read";
        return false;
    }
    u32 w = 0, h = 0;
    if (!ReadSurface(*surface, out, &w, &h)) {
        *error = "the read back failed";
        return false;
    }
    *width = w;
    *height = h;
    return true;
}

std::string Describe() {
    if (!g_.built) {
        return "no program yet";
    }
    std::string text = "the combiner program, " + std::to_string(g_.surfaces.size()) + " colour buffers, " +
                       std::to_string(g_.images.size()) + " textures";
    if (g_.current != nullptr) {
        text += ", the last " + std::to_string(g_.current->target.width) + "x" +
                std::to_string(g_.current->target.height);
    }
    return text;
}

void Forget() {
    if (!g_.built) {
        return;
    }
    for (const auto& [key, image] : g_.images) {
        GLuint gone = image;
        glDeleteTextures(1, &gone);
    }
    g_.images.clear();
    for (auto& [key, surface] : g_.surfaces) {
        DeleteSurface(surface);
    }
    g_.surfaces.clear();
    for (VertexProgram& stage : g_.programs) {
        GLuint gone = stage.program;
        glDeleteProgram(gone);
    }
    g_.programs.clear();
    for (GLuint* buffer : {&g_.program_vbo, &g_.program_ibo, &g_.program_ubo, &g_.program_data_ubo}) {
        if (*buffer != 0) {
            glDeleteBuffers(1, buffer);
            *buffer = 0;
        }
    }
    if (g_.program_vao != 0) {
        GLuint gone = g_.program_vao;
        glDeleteVertexArrays(1, &gone);
        g_.program_vao = 0;
    }
    GLuint vao = g_.vao, vbo = g_.vbo, program = g_.program, fog_lut = g_.fog_lut, proctex = g_.proctex;
    GLuint composite_vao = g_.composite.vao, composite_program = g_.composite.program;
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteTextures(1, &fog_lut);
    if (proctex != 0) {
        glDeleteTextures(1, &proctex);
    }
    glDeleteProgram(program);
    if (composite_vao != 0) {
        glDeleteVertexArrays(1, &composite_vao);
    }
    if (composite_program != 0) {
        glDeleteProgram(composite_program);
    }
    g_ = Objects{};
}

}  // namespace mh4n::gpu::gles::render

#else  // !__ANDROID__

namespace mh4n::gpu::gles::render {

namespace {
const char* const kNoGles = "the GLES renderer is built for Android only";
}

bool IsPassStarted(const Target&) {
    return false;
}

bool Begin(const Target&, const TargetContents*, std::string* error) {
    *error = kNoGles;
    return false;
}

void EndFrame(u32) {}

void EndPasses() {}

bool DrawTriangles(const void*, u32, const DrawState&, std::string* error) {
    *error = kNoGles;
    return false;
}

bool DrawVertexProgram(const VertexDraw&, const DrawState&, std::string* error) {
    *error = kNoGles;
    return false;
}

bool ReadTarget(u32, std::vector<u8>&, u32* width, u32* height, std::string* error) {
    *width = 0;
    *height = 0;
    *error = kNoGles;
    return false;
}

bool WriteBackDrawn(const StorePicture&) {
    return true;
}

std::string Describe() {
    return kNoGles;
}

void Forget() {}

}  // namespace mh4n::gpu::gles::render

#endif  // __ANDROID__
