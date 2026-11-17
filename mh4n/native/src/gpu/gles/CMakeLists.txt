# mh4n_gpu_gles: the alternative renderer (`gles`), on an OpenGL ES 3.1 context of its own instead of Vulkan. The
# module is built on every host (the PC runner and the tests link every module), but only Android has EGL and GLES:
# elsewhere gles_context.cpp compiles to the same functions answering "this host has none".
#
# The GX commands, the register file, the fills, the DMA and the interrupts are the null GPU's (mh4n_gpu_null) until the
# drawing lands on this path.
mh4n_module(
    SOURCES
        gles_context.cpp
        gles_gpu.cpp
    DEPENDS
        mh4n_gpu_null
)
# The header is included as "gpu/gles/gles_gpu.h".
target_include_directories(mh4n_gpu_gles PUBLIC "${MH4N_SRC_DIR}")
if (ANDROID)
    # libGLESv3 is where the ES 3 entry points are (the app links GLESv2 for SDL's renderer).
    target_link_libraries(mh4n_gpu_gles PUBLIC EGL GLESv3)
endif()
