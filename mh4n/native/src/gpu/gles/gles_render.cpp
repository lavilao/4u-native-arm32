// The GLES rasterizer (gles_render.h).
#include "gpu/gles/gles_render.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_map>

#include "gpu/cpuvertex/triangle_batch.h"
#include "gpu/gles/gles_context.h"
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
out vec4 v_colour;
out vec2 v_texcoord0;
out vec2 v_texcoord1;
out vec2 v_texcoord2;
void main() {
    gl_Position = a_position;
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

out vec4 o_colour;

// The PICA's eight bits of precision between stages (generator.cpp's byteround).
float byteround(float x) {
    return round(clamp(x, 0.0, 1.0) * 255.0) / 255.0;
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
// Texture3 6 (the procedural texture: not here), PreviousBuffer 0xD, Constant 0xE, Previous 0xF. The primary fragment
// colour is the lighting result, which this renderer does not shade: the vertex colour stands in for it, and the
// secondary colour is nothing.
vec4 TevSource(int source, vec4 primary, vec4 previous, vec4 held, vec4 constant_colour,
               vec4 texture0, vec4 texture1, vec4 texture2) {
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

void main() {
    vec4 primary = byteround(v_colour);
    vec4 texture0 = u_has_texture0 != 0 ? texture(u_texture0, Coordinate(u_coordinate0, v_texcoord0, v_texcoord1, v_texcoord2)) : vec4(0.0);
    vec4 texture1 = u_has_texture1 != 0 ? texture(u_texture1, Coordinate(u_coordinate1, v_texcoord0, v_texcoord1, v_texcoord2)) : vec4(0.0);
    vec4 texture2 = u_has_texture2 != 0 ? texture(u_texture2, Coordinate(u_coordinate2, v_texcoord0, v_texcoord1, v_texcoord2)) : vec4(0.0);

    vec4 previous = vec4(0.0);            // combiner_output
    vec4 held = vec4(0.0);                // combiner_buffer
    vec4 next_buffer = u_combiner_colour; // next_combiner_buffer
    for (int stage = 0; stage < 6; ++stage) {
        const int sources = u_source[stage];
        const int modifiers = u_modifier[stage];
        const int ops = u_ops[stage];
        const int scales = u_scale[stage];
        const vec4 constant_colour = u_constant[stage];
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
                                                texture0, texture1, texture2),
                                      modifier[i]);
        }
        const int color_op = ops & 0xF;
        const int alpha_op = (ops >> 16) & 0xF;
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
                                                    texture0, texture1, texture2),
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
    o_colour = previous;
}
)";

struct Uniforms {
    GLint source = -1, modifier = -1, ops = -1, scale = -1, constant = -1;
    GLint combiner_colour = -1, update_rgb = -1, update_alpha = -1;
    GLint alpha_test = -1, alpha_func = -1, alpha_ref = -1;
    GLint texture[3] = {-1, -1, -1}, has_texture[3] = {-1, -1, -1}, coordinate[3] = {-1, -1, -1};
};

struct Objects {
    bool built = false;
    bool started = false;  // a pass is in progress for `target` (Begin started it; EndFrame ends it)
    GLuint program = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint fbo = 0;
    GLuint colour = 0;  // the colour target, as a texture (so it can be read back)
    GLuint depth = 0;   // its depth renderbuffer
    Target target;
    Target storage;  // the size the attachments have
    Uniforms uniforms;
    std::unordered_map<u64, GLuint> images;  // a texture's key -> the GL image of its pixels
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
    Uniforms& u = g_.uniforms;
    u.source = glGetUniformLocation(program, "u_source[0]");
    u.modifier = glGetUniformLocation(program, "u_modifier[0]");
    u.ops = glGetUniformLocation(program, "u_ops[0]");
    u.scale = glGetUniformLocation(program, "u_scale[0]");
    u.constant = glGetUniformLocation(program, "u_constant[0]");
    u.combiner_colour = glGetUniformLocation(program, "u_combiner_colour");
    u.update_rgb = glGetUniformLocation(program, "u_update_rgb");
    u.update_alpha = glGetUniformLocation(program, "u_update_alpha");
    u.alpha_test = glGetUniformLocation(program, "u_alpha_test");
    u.alpha_func = glGetUniformLocation(program, "u_alpha_func");
    u.alpha_ref = glGetUniformLocation(program, "u_alpha_ref");
    for (int unit = 0; unit < 3; ++unit) {
        const std::string texture = "u_texture" + std::to_string(unit);
        const std::string has = "u_has_texture" + std::to_string(unit);
        const std::string coordinate = "u_coordinate" + std::to_string(unit);
        u.texture[unit] = glGetUniformLocation(program, texture.c_str());
        u.has_texture[unit] = glGetUniformLocation(program, has.c_str());
        u.coordinate[unit] = glGetUniformLocation(program, coordinate.c_str());
    }
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

// The colour target and its depth: the storage is made (again when its size changes) and the pass starts empty, as the
// game's first draw into it expects (it fills it).
bool MakeTarget(const Target& target, std::string* error) {
    if (g_.fbo == 0) {
        glGenFramebuffers(1, &g_.fbo);
    }
    if (g_.colour == 0) {
        glGenTextures(1, &g_.colour);
    }
    if (g_.depth == 0) {
        glGenRenderbuffers(1, &g_.depth);
    }
    if (!(g_.storage.width == target.width && g_.storage.height == target.height)) {
        glBindTexture(GL_TEXTURE_2D, g_.colour);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(target.width), static_cast<GLsizei>(target.height),
                     0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, g_.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_.colour, 0);
        // The depth buffer the 3D draws test against: 24 bits where the driver makes one, 16 where it does not.
        const GLenum depth_formats[] = {GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT16};
        GLenum status = GL_FRAMEBUFFER_UNSUPPORTED;
        for (const GLenum format : depth_formats) {
            glBindRenderbuffer(GL_RENDERBUFFER, g_.depth);
            glRenderbufferStorage(GL_RENDERBUFFER, format, static_cast<GLsizei>(target.width),
                                  static_cast<GLsizei>(target.height));
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_.depth);
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status == GL_FRAMEBUFFER_COMPLETE) {
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
        g_.storage = target;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, g_.fbo);
    glViewport(0, 0, static_cast<GLsizei>(target.width), static_cast<GLsizei>(target.height));
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepthf(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    g_.target = target;
    g_.started = true;
    return true;
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(texture.width), static_cast<GLsizei>(texture.height),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, texture.rgba8);
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

bool Begin(const Target& target, std::string* error) {
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
    // A draw into another framebuffer (the other screen, or the frame after a swap) starts its pass: the target's
    // picture is cleared for it.
    if (!g_.started || !(g_.target == target)) {
        if (!MakeTarget(target, error)) {
            return false;
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, g_.fbo);
    glViewport(0, 0, static_cast<GLsizei>(g_.storage.width), static_cast<GLsizei>(g_.storage.height));
    glDisable(GL_SCISSOR_TEST);
    return true;
}

void EndFrame() {
    g_.started = false;
}

bool DrawTriangles(const void* vertices, u32 count, const DrawState& state, std::string* error) {
    if (count == 0) {
        return true;
    }
    if (!g_.built || g_.fbo == 0) {
        *error = "the rasterizer has no target (Begin makes it)";
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, g_.fbo);
    glUseProgram(g_.program);
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

    const Uniforms& u = g_.uniforms;
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

    for (int unit = 0; unit < 3; ++unit) {
        const bool set = unit < state.texture_count && state.textures[unit].rgba8 != nullptr;
        glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
        if (set) {
            const Texture& texture = state.textures[unit];
            glBindTexture(GL_TEXTURE_2D, ImageOf(texture));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, WrapMode(texture.wrap_s));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, WrapMode(texture.wrap_t));
            glUniform1i(u.texture[unit], unit);
        } else {
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glUniform1i(u.has_texture[unit], set ? 1 : 0);
        glUniform1i(u.coordinate[unit], static_cast<GLint>(set ? std::min(state.textures[unit].coordinate, 2u) : 0u));
    }

    glBindVertexArray(g_.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(static_cast<std::size_t>(count) * kVertexStride), vertices,
                 GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(count));
    const GLenum error_gl = glGetError();
    if (error_gl != GL_NO_ERROR) {
        char why[96];
        std::snprintf(why, sizeof(why), "the draw failed (0x%04X)", static_cast<unsigned>(error_gl));
        *error = why;
        return false;
    }
    return true;
}

bool ReadTarget(std::vector<u8>& out, u32* width, u32* height, std::string* error) {
    if (!g_.built || g_.fbo == 0) {
        *error = "there is no target to read";
        return false;
    }
    const u32 w = g_.storage.width, h = g_.storage.height;
    out.resize(static_cast<std::size_t>(w) * h * 4);
    glBindFramebuffer(GL_FRAMEBUFFER, g_.fbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h), GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    const GLenum error_gl = glGetError();
    if (error_gl != GL_NO_ERROR) {
        char why[96];
        std::snprintf(why, sizeof(why), "the read back failed (0x%04X)", static_cast<unsigned>(error_gl));
        *error = why;
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

std::string Describe() {
    if (!g_.built) {
        return "no program yet";
    }
    return "the combiner program, a " + std::to_string(g_.storage.width) + "x" + std::to_string(g_.storage.height) +
           " target, " + std::to_string(g_.images.size()) + " textures";
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
    GLuint vao = g_.vao, vbo = g_.vbo, fbo = g_.fbo, colour = g_.colour, depth = g_.depth, program = g_.program;
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &colour);
    glDeleteRenderbuffers(1, &depth);
    glDeleteProgram(program);
    g_ = Objects{};
}

}  // namespace mh4n::gpu::gles::render

#else  // !__ANDROID__

namespace mh4n::gpu::gles::render {

namespace {
const char* const kNoGles = "the GLES renderer is built for Android only";
}

bool Begin(const Target&, std::string* error) {
    *error = kNoGles;
    return false;
}

void EndFrame() {}

bool DrawTriangles(const void*, u32, const DrawState&, std::string* error) {
    *error = kNoGles;
    return false;
}

bool ReadTarget(std::vector<u8>&, u32* width, u32* height, std::string* error) {
    *width = 0;
    *height = 0;
    *error = kNoGles;
    return false;
}

std::string Describe() {
    return kNoGles;
}

void Forget() {}

}  // namespace mh4n::gpu::gles::render

#endif  // __ANDROID__
