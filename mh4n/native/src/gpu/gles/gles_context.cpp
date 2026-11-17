// The GLES 3.1 context of the alternative renderer (gles_context.h).
#include "gpu/gles/gles_context.h"

#include "mh4n/log.h"

#if defined(__ANDROID__)

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <cstdio>

namespace mh4n::gpu::gles {

namespace {

constexpr const char* kLog = "GlesGpu";

// EGL_CONTEXT_CLIENT_VERSION and EGL_CONTEXT_MINOR_VERSION_KHR (eglext.h's value): the attributes of an ES 3.1
// context, spelled out so that this file needs no extension header for them.
constexpr EGLint kContextClientVersion = 0x3098;
constexpr EGLint kContextMinorVersion = 0x30FB;

EGLDisplay g_display = EGL_NO_DISPLAY;
EGLContext g_context = EGL_NO_CONTEXT;
EGLSurface g_surface = EGL_NO_SURFACE;
ContextInfo g_info;
std::string g_error;
bool g_tried = false;

std::string StringOf(GLenum name) {
    const GLubyte* text = glGetString(name);
    return text != nullptr ? std::string(reinterpret_cast<const char*>(text)) : std::string();
}

std::string StringOfDisplay(EGLDisplay display, EGLint name) {
    const char* text = eglQueryString(display, name);
    return text != nullptr ? std::string(text) : std::string();
}

// An ES 3 config with the formats the game's framebuffers use (RGBA8, D24S8) and no window: the screens are
// framebuffer objects of the renderer's own.
EGLConfig ChooseConfig(EGLDisplay display) {
    const EGLint wanted[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                             EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                             EGL_RED_SIZE, 8,
                             EGL_GREEN_SIZE, 8,
                             EGL_BLUE_SIZE, 8,
                             EGL_ALPHA_SIZE, 8,
                             EGL_DEPTH_SIZE, 24,
                             EGL_STENCIL_SIZE, 8,
                             EGL_NONE};
    EGLConfig config = nullptr;
    EGLint count = 0;
    if (eglChooseConfig(display, wanted, &config, 1, &count) != EGL_TRUE || count < 1) {
        // A driver without a 24-bit depth config still draws: ask only for what no ES3 config lacks.
        const EGLint loose[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                                EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
        count = 0;
        if (eglChooseConfig(display, loose, &config, 1, &count) != EGL_TRUE || count < 1) {
            return nullptr;
        }
    }
    return config;
}

// The context as asked for (ES 3.1, for compute and the storage buffers the renderer's shaders use), else the best the
// driver offers (ES 3.0), which is logged.
EGLContext CreateContext(EGLDisplay display, EGLConfig config, std::string* how) {
    const EGLint es31[] = {kContextClientVersion, 3, kContextMinorVersion, 1, EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, es31);
    if (context != EGL_NO_CONTEXT) {
        *how = "ES 3.1";
        return context;
    }
    const EGLint es30[] = {kContextClientVersion, 3, EGL_NONE};
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, es30);
    if (context != EGL_NO_CONTEXT) {
        *how = "ES 3.0 (no 3.1 context)";
    }
    return context;
}

}  // namespace

bool MakeCurrent(std::string* error) {
    auto fail = [error](const std::string& why) {
        g_error = why;
        if (error != nullptr) {
            *error = why;
        }
        MH4N_LOG(Error, kLog, "no GLES context: {}", why);
        return false;
    };
    if (g_context != EGL_NO_CONTEXT) {
        if (eglMakeCurrent(g_display, g_surface, g_surface, g_context) == EGL_TRUE) {
            return true;
        }
        return fail("this thread could not make the context current");
    }
    if (g_tried) {
        if (error != nullptr) {
            *error = g_error;
        }
        return false;
    }
    g_tried = true;

    g_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (g_display == EGL_NO_DISPLAY) {
        return fail("eglGetDisplay gave no display");
    }
    EGLint major = 0, minor = 0;
    if (eglInitialize(g_display, &major, &minor) != EGL_TRUE) {
        char why[64];
        std::snprintf(why, sizeof(why), "eglInitialize failed (0x%04X)", static_cast<unsigned>(eglGetError()));
        return fail(why);
    }
    if (eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE) {
        return fail("eglBindAPI(EGL_OPENGL_ES_API) failed");
    }
    const EGLConfig config = ChooseConfig(g_display);
    if (config == nullptr) {
        return fail("the driver offers no OpenGL ES 3 config");
    }
    std::string how;
    g_context = CreateContext(g_display, config, &how);
    if (g_context == EGL_NO_CONTEXT) {
        char why[64];
        std::snprintf(why, sizeof(why), "eglCreateContext failed (0x%04X)", static_cast<unsigned>(eglGetError()));
        return fail(why);
    }
    const EGLint pbuffer[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    g_surface = eglCreatePbufferSurface(g_display, config, pbuffer);
    if (g_surface == EGL_NO_SURFACE) {
        char why[96];
        std::snprintf(why, sizeof(why), "eglCreatePbufferSurface failed (0x%04X)", static_cast<unsigned>(eglGetError()));
        return fail(why);
    }
    if (eglMakeCurrent(g_display, g_surface, g_surface, g_context) != EGL_TRUE) {
        char why[64];
        std::snprintf(why, sizeof(why), "eglMakeCurrent failed (0x%04X)", static_cast<unsigned>(eglGetError()));
        return fail(why);
    }

    g_info.gl_version = StringOf(GL_VERSION);
    g_info.gl_renderer = StringOf(GL_RENDERER);
    g_info.gl_slang = StringOf(GL_SHADING_LANGUAGE_VERSION);
    g_info.egl_version = StringOfDisplay(g_display, EGL_VERSION);
    g_info.egl_vendor = StringOfDisplay(g_display, EGL_VENDOR);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &g_info.max_texture);
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &g_info.max_renderbuffer);
    MH4N_LOG(Info, kLog, "gl: {} on \"{}\", GLSL \"{}\"; EGL \"{}\", vendor \"{}\"", g_info.gl_version, g_info.gl_renderer,
             g_info.gl_slang, g_info.egl_version, g_info.egl_vendor);
    MH4N_LOG(Info, kLog, "gl: {}", how);
    return true;
}

bool Available() {
    return g_context != EGL_NO_CONTEXT;
}

const ContextInfo& Info() {
    return g_info;
}

std::string Describe() {
    if (g_context == EGL_NO_CONTEXT) {
        return g_tried ? "no GLES context (" + g_error + ")" : "no GLES context made";
    }
    return g_info.gl_version + " on \"" + g_info.gl_renderer + "\", EGL " + g_info.egl_version +
           " (textures up to " + std::to_string(g_info.max_texture) + ")";
}

}  // namespace mh4n::gpu::gles

#else  // !__ANDROID__

namespace mh4n::gpu::gles {

bool MakeCurrent(std::string* error) {
    if (error != nullptr) {
        *error = "the GLES renderer is built for Android only";
    }
    return false;
}

bool Available() {
    return false;
}

const ContextInfo& Info() {
    static const ContextInfo none;
    return none;
}

std::string Describe() {
    return "no GLES context (this host has none: the GLES renderer is built for Android only)";
}

}  // namespace mh4n::gpu::gles

#endif  // __ANDROID__
