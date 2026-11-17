# mh4n_gpu_gles: the alternative renderer (`gles`), on an OpenGL ES 3.1 context of its own instead of Vulkan. The
# module is built on every host (the PC runner and the tests link every module), but only Android has EGL and GLES:
# elsewhere gles_context.cpp and gles_render.cpp compile to the same functions answering "this host has none".
#
# What it is made of: the GX engine (mh4n_gpu_gx) for the memory work, the PICA command processor (mh4n_gpu_pica) for the
# command lists, the CPU vertex path (mh4n_gpu_cpuvertex) for the transform and shading, the texture cache
# (mh4n_gpu_texture) for the images, and gles_render.cpp for the pixels.
mh4n_module(
    SOURCES
        gles_context.cpp
        gles_draw.cpp
        gles_gpu.cpp
        gles_render.cpp
    DEPENDS
        mh4n_gpu_cpuvertex
        mh4n_gpu_gx
        mh4n_gpu_pica
        mh4n_gpu_texture
)
# The headers are included as "gpu/gles/<name>.h".
target_include_directories(mh4n_gpu_gles PUBLIC "${MH4N_SRC_DIR}")
if (ANDROID)
    # libGLESv3 is where the ES 3 entry points are (the app links GLESv2 for SDL's renderer).
    target_link_libraries(mh4n_gpu_gles PUBLIC EGL GLESv3)
endif()
