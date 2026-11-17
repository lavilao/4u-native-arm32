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
// 5 texcoord0_w, 6 normquat, 7 view. The trivial stage here reads the first three.
constexpr unsigned kVertexStride = 88;
constexpr unsigned kAttribCount = 8;
// The images kept for the textures drawn: past this many, the least useful go (a new pass uploads them again).
constexpr std::size_t kMaxImages = 192;

// The vertex stage: the batch is clip space already (the CPU placed and shaded it), so it goes straight out, and what the
// fragment stage needs goes with it.
constexpr const char* kVertexSource = R"(#version 310 es
layout(location = 0) in vec4 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_texcoord0;
out vec4 v_color;
out vec2 v_texcoord0;
void main() {
    gl_Position = a_position;
    gl_PointSize = 1.0;
    v_color = a_color;
    v_texcoord0 = a_texcoord0;
}
)";

// The fragment stage: the vertex colour, times the first texture when the draw has one. The combiner, the fog, the alpha
// test and the output merger are not here yet (gles_render.h says so).
constexpr const char* kFragmentSource = R"(#version 310 es
precision mediump float;
in vec4 v_color;
in vec2 v_texcoord0;
uniform sampler2D u_texture;
uniform int u_has_texture;
out vec4 o_color;
void main() {
    vec4 color = v_color;
    if (u_has_texture != 0) {
        color *= texture(u_texture, v_texcoord0);
    }
    o_color = color;
}
)";

struct Objects {
    bool built = false;
    bool cleared = false;  // the current target has been cleared for its pass
    GLuint program = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint fbo = 0;
    GLuint colour = 0;  // the colour target, as a texture (so it can be read back)
    GLuint depth = 0;   // its depth renderbuffer
    Target target;
    GLint has_texture = -1;
    GLint u_texture = -1;
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
    g_.has_texture = glGetUniformLocation(program, "u_has_texture");
    g_.u_texture = glGetUniformLocation(program, "u_texture");
    glGenVertexArrays(1, &g_.vao);
    glGenBuffers(1, &g_.vbo);
    glBindVertexArray(g_.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_.vbo);
    // The layout is fixed (kVertexStride and the offsets of HardwareVertex); the locations the stage does not read are
    // described too, so the array is complete for a stage that comes to read them (6 normquat, 7 view).
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
    GLenum error_gl = glGetError();
    if (error_gl != GL_NO_ERROR) {
        char why[96];
        std::snprintf(why, sizeof(why), "the vertex array failed (0x%04X)", static_cast<unsigned>(error_gl));
        *error = why;
        return false;
    }
    g_.built = true;
    return true;
}

// The colour target and its depth, made for `target` (again when the size changes).
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
    glBindTexture(GL_TEXTURE_2D, g_.colour);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(target.width), static_cast<GLsizei>(target.height), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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
    g_.target = target;
    g_.cleared = false;
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    g_.images.emplace(texture.key, image);
    return image;
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
    const bool made = !(g_.target == target) || g_.cleared == false || g_.fbo == 0;
    if (!(g_.target == target) || g_.fbo == 0) {
        if (!MakeTarget(target, error)) {
            return false;
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, g_.fbo);
    glViewport(0, 0, static_cast<GLsizei>(g_.target.width), static_cast<GLsizei>(g_.target.height));
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    if (made) {
        // A new target is a new pass: the game's first draw into it fills it, and this renderer clears it first so that
        // what the last pass left is not shown (the PICA's own clear draws come through as draws for now).
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClearDepthf(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        g_.cleared = true;
    }
    return true;
}

bool DrawTriangles(const void* vertices, u32 count, const DrawState& state, const Texture* texture, std::string* error) {
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
    glDepthMask(state.depth_write ? GL_TRUE : GL_FALSE);
    int has_texture = 0;
    if (texture != nullptr && texture->rgba8 != nullptr && texture->width != 0 && texture->height != 0) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ImageOf(*texture));
        glUniform1i(g_.u_texture, 0);
        has_texture = 1;
    }
    glUniform1i(g_.has_texture, has_texture);
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
    const u32 w = g_.target.width, h = g_.target.height;
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
    return "the pass-through program, a " + std::to_string(g_.target.width) + "x" + std::to_string(g_.target.height) +
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

bool DrawTriangles(const void*, u32, const DrawState&, const Texture*, std::string* error) {
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
