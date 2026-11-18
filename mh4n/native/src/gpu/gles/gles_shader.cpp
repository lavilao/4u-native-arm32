// The Vulkan vertex stage's GLSL as a GLSL ES 3.10 stage (gles_shader.h).
#include "gpu/gles/gles_shader.h"

namespace mh4n::gpu::gles {

namespace {

// The line that ends the varying named `name` (the generator writes one line for each of them).
constexpr const char* kVaryingLines[] = {
    "layout(location = 1) out vec4 primary_color;\n",
    "layout(location = 2) out vec2 texcoord0;\n",
    "layout(location = 3) out vec2 texcoord1;\n",
    "layout(location = 4) out vec2 texcoord2;\n",
    "layout(location = 5) out float texcoord0_w;\n",
    "layout(location = 6) out vec4 normquat;\n",
    "layout(location = 7) out vec3 view;\n",
};

}  // namespace

bool ToEsVertexStage(const std::string& vulkan_glsl, std::string* out, std::string* error) {
    auto fail = [error](const char* why) {
        *error = why;
        return false;
    };
    std::string s = vulkan_glsl;
    if (s.rfind("#version 450\n", 0) != 0) {
        return fail("the source is not the #version 450 stage this converts");
    }
    // (ES needs both precisions declared; the driver then has no default to invent)
    s.replace(0, std::string("#version 450\n").size(),
              "#version 310 es\nprecision highp float;\nprecision highp int;\n");

    // The uniform blocks by index (the renderer binds them: gles_render.cpp).
    for (;;) {
        const std::size_t at = s.find("layout (set = 0, binding = ");
        if (at == std::string::npos) {
            break;
        }
        const std::size_t end = s.find(", std140)", at);
        if (end == std::string::npos) {
            return fail("a uniform block's layout does not end in std140");
        }
        s.replace(at, end + std::string(", std140)").size() - at, "layout (std140)");
    }
    if (s.find("set = 0") != std::string::npos) {
        return fail("the source has a descriptor set this does not convert");
    }
    // ES has no gl_ClipDistance: the stage must have been generated without it (config.use_clip_planes false). Its
    // `out_clip` pair of values is generated either way and goes with the writes below.
    if (s.find("gl_ClipDistance") != std::string::npos) {
        return fail("the stage clips with gl_ClipDistance, which GLSL ES has not");
    }

    // The varyings the fragment program declares, by name (ES 3.0 links them without locations).
    const char* const kVaryings[] = {"out vec4 v_colour;\n", "out vec2 v_texcoord0;\n", "out vec2 v_texcoord1;\n",
                                     "out vec2 v_texcoord2;\n", "", "", ""};
    for (std::size_t i = 0; i < 7; ++i) {
        const std::size_t at = s.find(kVaryingLines[i]);
        if (at == std::string::npos) {
            return fail("the stage's varyings are not the seven the generator writes");
        }
        s.replace(at, std::string(kVaryingLines[i]).size(), kVaryings[i]);
    }

    // The same depth Vulkan draws with, carried through the ES's clip space (see gles_shader.h).
    constexpr const char* kVulkanPosition = "vec4 out_position = vec4(vtx_pos.x, vtx_pos.y, -vtx_pos.z, vtx_pos.w);";
    constexpr const char* kEsPosition =
        "vec4 out_position = vec4(vtx_pos.x, vtx_pos.y, -2.0 * vtx_pos.z - vtx_pos.w, vtx_pos.w);";
    const std::size_t position_at = s.find(kVulkanPosition);
    if (position_at == std::string::npos) {
        return fail("the stage's position is not the one the generator writes");
    }
    s.replace(position_at, std::string(kVulkanPosition).size(), kEsPosition);

    // The assignments of the varyings that are gone, and the names of the ones that stay.
    const std::pair<const char*, const char*> kAssignments[] = {
        {"    normquat = GetVertexQuaternion();\n", ""},
        {"    primary_color = ", "    v_colour = "},
        {"    texcoord0 = vec2(", "    v_texcoord0 = vec2("},
        {"    texcoord1 = vec2(", "    v_texcoord1 = vec2("},
        {"    texcoord2 = vec2(", "    v_texcoord2 = vec2("},
        {"    texcoord0_w = ", ""},
        {"    view = ", ""},
    };
    for (const auto& [from, to] : kAssignments) {
        const std::size_t at = s.find(from);
        if (at == std::string::npos) {
            return fail("the stage's EmitVtx is not the one the generator writes");
        }
        const std::size_t line_end = s.find('\n', at);
        if (line_end == std::string::npos) {
            return fail("the stage's EmitVtx has no last line");
        }
        // (texcoord0_w's and view's lines go whole; the others keep what follows the name)
        s.replace(at, std::string(to).empty() ? line_end + 1 - at : std::string(from).size(), to);
    }
    if (s.find("normquat") != std::string::npos || s.find("texcoord0_w") != std::string::npos ||
        s.find("\n    view = ") != std::string::npos) {
        return fail("the stage writes a varying the renderer's fragment program has not");
    }

    *out = std::move(s);
    return true;
}

}  // namespace mh4n::gpu::gles
