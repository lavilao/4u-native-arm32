// The OpenGL ES 3.1 context the alternative renderer draws with (gpu/gles).
//
// A context of its own, with no window: Android's drivers want a surface, so the context is made current with a 1x1
// pbuffer, and everything the renderer draws goes into framebuffer objects of its own. The context belongs to the
// thread that called MakeCurrent (an EGL context is current on one thread at a time): the GPU's thread makes it current
// there, and a renderer thread of its own makes it current on that one.
//
// Built for Android only: on another host MakeCurrent fails with the reason, so the `gles` GPU stays the null GPU's
// command processing and nothing tries to call GL.
#pragma once

#include <string>

namespace mh4n::gpu::gles {

// What the driver said it is, once MakeCurrent has succeeded (empty before).
struct ContextInfo {
    std::string gl_version;   // GL_VERSION
    std::string gl_renderer;  // GL_RENDERER
    std::string gl_slang;     // GL_SHADING_LANGUAGE_VERSION
    std::string egl_version;  // EGL_VERSION
    std::string egl_vendor;   // EGL_VENDOR
    int max_texture = 0;      // GL_MAX_TEXTURE_SIZE
    int max_renderbuffer = 0; // GL_MAX_RENDERBUFFER_SIZE
};

// Makes the process's GLES context current on the calling thread, making it on the first call. False, and `error` set,
// when this host has no GLES 3 context (a PC build, or a driver that refused one); the reason is logged once.
bool MakeCurrent(std::string* error);

// Whether a context has been made (MakeCurrent succeeded at least once).
bool Available();

// The driver's strings and limits; empty/zero until MakeCurrent has succeeded.
const ContextInfo& Info();

// One line for log.txt: the driver as the Vulkan renderer logs its device.
std::string Describe();

}  // namespace mh4n::gpu::gles
