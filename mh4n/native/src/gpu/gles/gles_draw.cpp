// The PICA's draws on the GLES rasterizer (gles_draw.h).
#include "gpu/gles/gles_draw.h"

#include <algorithm>
#include <cstring>

#include "gpu/pica/regs.h"
#include "mh4n/log.h"

namespace mh4n::gpu::gles {

namespace {

constexpr const char* kLog = "GlesGpu";

// RegsInternal: FramebufferRegs at 0x100, its framebuffer block at 0x110; RasterizerRegs at 0x40 (as native_gpu.cpp
// and fbops/fixed_function.cpp read them).
constexpr u32 kRegCullMode = 0x40;
constexpr u32 kRegViewportSizeX = 0x41;
constexpr u32 kRegViewportSizeY = 0x43;
constexpr u32 kRegScissorXY1 = 0x66;
constexpr u32 kRegScissorXY2 = 0x67;
constexpr u32 kRegViewportCorner = 0x68;
constexpr u32 kRegMainConfig = 0x80;
constexpr u32 kRegTextureUnit[3] = {0x81, 0x91, 0x99};  // a unit: border colour, size, parameters, lod, address
constexpr u32 kTextureParams = 2;                        // wrap t 8-10, wrap s 12-14
constexpr u32 kRegTevStage[6] = {0xC0, 0xC8, 0xD0, 0xD8, 0xF0, 0xF8};
constexpr u32 kTevSources = 0, kTevModifiers = 1, kTevOps = 2, kTevConstColour = 3, kTevScales = 4;
constexpr u32 kRegFogModeAndMasks = 0xE0;  // fog mode 0-2, update mask rgb 8-11, update mask alpha 12-15, fog flip 16
constexpr u32 kRegFogColour = 0xE1;       // r 0-7, g 8-15, b 16-23
constexpr u32 kRegTevCombinerBuffer = 0xFD;
constexpr u32 kRegOutputMerger = 0x100;  // fragment operation mode 0-1, alphablend enable 8
constexpr u32 kRegLogicOp = 0x102;        // bits 0-3 (ignored while blending is on)
constexpr u32 kRegBlendEquations = 0x101;
constexpr u32 kRegBlendConst = 0x103;
constexpr u32 kRegAlphaTest = 0x104;      // enable 0, function 4-6, reference 8-15
constexpr u32 kRegDepthColourMask = 0x107;  // depth test 0, function 4-6, colour mask 8-11, depth write 12
constexpr u32 kRegAllowColourWrite = 0x113;  // bits 0-3
constexpr u32 kRegAllowDepthStencil = 0x115;  // bits 0-1
constexpr u32 kRegColourFormat = 0x117;
constexpr u32 kRegColourAddress = 0x11D;  // bits 0-27, in units of 8 bytes
constexpr u32 kRegFramebufferSize = 0x11E;

constexpr u32 kGasDensityMode = 1;  // FragmentOperationMode::Gas (the fog's density pass: no colour)
constexpr u32 kShadowMode = 3;      // FragmentOperationMode::Shadow (into the shadow map: no colour)

// Azahar's Common::Rectangle: top >= bottom (y up).
struct Rect4 {
    s32 left = 0, top = 0, right = 0, bottom = 0;
};

s32 Signed10(u32 v) {
    return static_cast<s32>(v << 22) >> 22;
}

// The viewport and scissor of a draw, in GL's pixels: the framebuffer's y axis points down there, so a rectangle whose
// top is `top` starts at `height - top` (MakeGeometry of native_gpu.cpp, read into GL).
void MakeRectangles(const std::array<u32, pica::kNumRegs>& regs, u32 height, bool flipped, render::DrawState* out) {
    const u32 corner_word = regs[kRegViewportCorner];
    const s32 corner_x = Signed10(pica::BitsOf(corner_word, 0, 10));
    const s32 corner_y = Signed10(pica::BitsOf(corner_word, 16, 10));
    const s32 size_x = static_cast<s32>(pica::Float24ToFloat(pica::BitsOf(regs[kRegViewportSizeX], 0, 24)) * 2);
    const s32 size_y = static_cast<s32>(pica::Float24ToFloat(pica::BitsOf(regs[kRegViewportSizeY], 0, 24)) * 2);
    // GetViewportRect: left, top = corner.y + height, right, bottom = corner.y
    Rect4 viewport{corner_x, corner_y + size_y, corner_x + size_x, corner_y};
    const s32 h = static_cast<s32>(height);
    if (flipped) {
        viewport = Rect4{viewport.left, h - viewport.bottom, viewport.right, h - viewport.top};
    }
    const s32 w = static_cast<s32>(regs[kRegFramebufferSize] & 0x7FF);
    Rect4 draw{std::clamp(viewport.left, 0, w), std::clamp(viewport.top, 0, h), std::clamp(viewport.right, 0, w),
               std::clamp(viewport.bottom, 0, h)};
    out->viewport_x = static_cast<float>(draw.left);
    out->viewport_y = static_cast<float>(h - draw.top);
    out->viewport_width = static_cast<float>(std::abs(draw.right - draw.left));
    out->viewport_height = static_cast<float>(std::abs(draw.top - draw.bottom));

    // The scissor registers: (x1, y1) in xy1, (x2, y2) in xy2, both inclusive, y up.
    Rect4 scissor{Signed10(pica::BitsOf(regs[kRegScissorXY1], 0, 10)),
                  static_cast<s32>(pica::BitsOf(regs[kRegScissorXY2], 16, 10)) + 1,
                  static_cast<s32>(pica::BitsOf(regs[kRegScissorXY2], 0, 10)) + 1,
                  Signed10(pica::BitsOf(regs[kRegScissorXY1], 16, 10))};
    if (flipped) {
        scissor = Rect4{scissor.left, h - scissor.bottom, scissor.right, h - scissor.top};
    }
    const s32 x0 = std::max(draw.left, scissor.left), x1 = std::min(draw.right, scissor.right);
    const s32 y0 = std::max(draw.bottom, scissor.bottom), y1 = std::min(draw.top, scissor.top);
    out->scissor_x = x0;
    out->scissor_y = h - y1;
    out->scissor_width = static_cast<u32>(std::max(x1 - x0, 0));
    out->scissor_height = static_cast<u32>(std::max(y1 - y0, 0));
}

// The output merger's state, as fbops/fixed_function.cpp (MakeFixedFunctionState) reads it, plus the culling.
void MakeFixedFunction(const std::array<u32, pica::kNumRegs>& regs, bool flipped, render::DrawState* out) {
    const u32 cull = pica::BitsOf(regs[kRegCullMode], 0, 2);
    out->cull = cull;
    out->flipped = flipped;

    const u32 merger = regs[kRegOutputMerger];
    const bool blending_enabled = pica::BitsOf(merger, 8, 1) != 0;
    // The canonical rule (gpu/fragment/key.cpp): the logic op register is the op only while blending is off; blending on
    // means the blender's result is what is written.
    out->logic_op = blending_enabled ? 3u : pica::BitsOf(regs[kRegLogicOp], 0, 4);
    out->blend = blending_enabled;
    if (out->blend) {
        const u32 word = regs[kRegBlendEquations];
        out->blend_equation_rgb = pica::BitsOf(word, 0, 3);
        out->blend_equation_alpha = pica::BitsOf(word, 8, 3);
        out->blend_source_rgb = pica::BitsOf(word, 16, 4);
        out->blend_dest_rgb = pica::BitsOf(word, 20, 4);
        out->blend_source_alpha = pica::BitsOf(word, 24, 4);
        out->blend_dest_alpha = pica::BitsOf(word, 28, 4);
        // A blend that writes the source as it is (One, Zero, Add in both halves; MH4U's opaque draws) is no blend.
        if (out->blend_source_rgb == 1 && out->blend_dest_rgb == 0 && out->blend_equation_rgb == 0 &&
            out->blend_source_alpha == 1 && out->blend_dest_alpha == 0 && out->blend_equation_alpha == 0) {
            out->blend = false;
        }
    }
    const u32 constant = regs[kRegBlendConst];
    for (u32 channel = 0; channel < 4; ++channel) {
        out->blend_constant[channel] = static_cast<float>(pica::BitsOf(constant, 8 * channel, 8)) / 255.0f;
    }

    const u32 depth_colour = regs[kRegDepthColourMask];
    out->colour_write = pica::BitsOf(regs[kRegAllowColourWrite], 0, 4) != 0
                            ? static_cast<u8>(pica::BitsOf(depth_colour, 8, 4))
                            : 0;
    const bool depth_test = pica::BitsOf(depth_colour, 0, 1) != 0;
    const bool depth_write = pica::BitsOf(depth_colour, 12, 1) != 0;
    if (depth_test || depth_write) {
        out->depth_test = true;
        out->depth_func = depth_test ? pica::BitsOf(depth_colour, 4, 3) : 1;  // Always
        out->depth_write = pica::BitsOf(regs[kRegAllowDepthStencil], 0, 2) != 0 && depth_write;
    } else {
        out->depth_test = false;
        out->depth_write = false;
    }

    // The alpha test, on the combiner's output alpha.
    out->fragment.alpha_test = pica::BitsOf(regs[kRegAlphaTest], 0, 1) != 0;
    out->fragment.alpha_func = pica::BitsOf(regs[kRegAlphaTest], 4, 3);
    out->fragment.alpha_ref = pica::BitsOf(regs[kRegAlphaTest], 8, 8);
}

// The six combiner stages and the buffer they hand on.
void MakeCombiner(const std::array<u32, pica::kNumRegs>& regs, render::FragmentState* out) {
    for (u32 stage = 0; stage < 6; ++stage) {
        const u32 base = kRegTevStage[stage];
        render::CombinerStage& s = out->stages[stage];
        s.sources = regs[base + kTevSources];
        s.modifiers = regs[base + kTevModifiers];
        s.ops = regs[base + kTevOps];
        s.scales = regs[base + kTevScales];
        s.constant = regs[base + kTevConstColour];
    }
    out->combiner_buffer_colour = regs[kRegTevCombinerBuffer];
    // The fog register's update masks: which of stages 0-3 write their output into the combiner buffer.
    out->update_rgb = pica::BitsOf(regs[kRegFogModeAndMasks], 8, 4);
    out->update_alpha = pica::BitsOf(regs[kRegFogModeAndMasks], 12, 4);
}

}  // namespace

u32 PicaBytesPerPixel(u32 format) {
    switch (format & 7) {
    case 1: return 3;  // RGB8
    case 2:            // RGB5A1
    case 3:            // RGB565
    case 4: return 2;  // RGBA4
    default: return 4; // RGBA8
    }
}

void EncodeColourRow(const u8* rgba8, u32 width, u32 format, u8* out) {
    switch (format & 7) {
    case 1:  // RGB8
        for (u32 x = 0; x < width; ++x) {
            out[x * 3 + 0] = rgba8[x * 4 + 0];
            out[x * 3 + 1] = rgba8[x * 4 + 1];
            out[x * 3 + 2] = rgba8[x * 4 + 2];
        }
        return;
    case 2:  // RGB5A1: rrrrrgggggbbbbba
        for (u32 x = 0; x < width; ++x) {
            const u16 r = static_cast<u16>((rgba8[x * 4 + 0] * 31u + 127u) / 255u);
            const u16 g = static_cast<u16>((rgba8[x * 4 + 1] * 31u + 127u) / 255u);
            const u16 b = static_cast<u16>((rgba8[x * 4 + 2] * 31u + 127u) / 255u);
            const u16 a = static_cast<u16>(rgba8[x * 4 + 3] >= 128 ? 1 : 0);
            const u16 value = static_cast<u16>((r << 11) | (g << 6) | (b << 1) | a);
            std::memcpy(out + x * 2, &value, 2);
        }
        return;
    case 3:  // RGB565: rrrrrggggggbbbbb
        for (u32 x = 0; x < width; ++x) {
            const u16 r = static_cast<u16>((rgba8[x * 4 + 0] * 31u + 127u) / 255u);
            const u16 g = static_cast<u16>((rgba8[x * 4 + 1] * 63u + 127u) / 255u);
            const u16 b = static_cast<u16>((rgba8[x * 4 + 2] * 31u + 127u) / 255u);
            const u16 value = static_cast<u16>((r << 11) | (g << 5) | b);
            std::memcpy(out + x * 2, &value, 2);
        }
        return;
    case 4:  // RGBA4: rrrrggggbbbbaaaa
        for (u32 x = 0; x < width; ++x) {
            const u16 r = static_cast<u16>((rgba8[x * 4 + 0] * 15u + 127u) / 255u);
            const u16 g = static_cast<u16>((rgba8[x * 4 + 1] * 15u + 127u) / 255u);
            const u16 b = static_cast<u16>((rgba8[x * 4 + 2] * 15u + 127u) / 255u);
            const u16 a = static_cast<u16>((rgba8[x * 4 + 3] * 15u + 127u) / 255u);
            const u16 value = static_cast<u16>((r << 12) | (g << 8) | (b << 4) | a);
            std::memcpy(out + x * 2, &value, 2);
        }
        return;
    default:  // RGBA8
        std::memcpy(out, rgba8, static_cast<std::size_t>(width) * 4);
        return;
    }
}

void DecodeColourRow(const u8* src, u32 width, u32 format, u8* rgba8) {
    switch (format & 7) {
    case 1:  // RGB8
        for (u32 x = 0; x < width; ++x) {
            rgba8[x * 4 + 0] = src[x * 3 + 0];
            rgba8[x * 4 + 1] = src[x * 3 + 1];
            rgba8[x * 4 + 2] = src[x * 3 + 2];
            rgba8[x * 4 + 3] = 255;
        }
        return;
    case 2:  // RGB5A1: rrrrrgggggbbbbba
        for (u32 x = 0; x < width; ++x) {
            u16 value = 0;
            std::memcpy(&value, src + x * 2, 2);
            const u16 r = static_cast<u16>((value >> 11) & 0x1F);
            const u16 g = static_cast<u16>((value >> 6) & 0x1F);
            const u16 b = static_cast<u16>((value >> 1) & 0x1F);
            rgba8[x * 4 + 0] = static_cast<u8>((r * 255u + 15u) / 31u);
            rgba8[x * 4 + 1] = static_cast<u8>((g * 255u + 15u) / 31u);
            rgba8[x * 4 + 2] = static_cast<u8>((b * 255u + 15u) / 31u);
            rgba8[x * 4 + 3] = (value & 1) != 0 ? 255 : 0;
        }
        return;
    case 3:  // RGB565: rrrrrggggggbbbbb
        for (u32 x = 0; x < width; ++x) {
            u16 value = 0;
            std::memcpy(&value, src + x * 2, 2);
            const u16 r = static_cast<u16>((value >> 11) & 0x1F);
            const u16 g = static_cast<u16>((value >> 5) & 0x3F);
            const u16 b = static_cast<u16>(value & 0x1F);
            rgba8[x * 4 + 0] = static_cast<u8>((r * 255u + 15u) / 31u);
            rgba8[x * 4 + 1] = static_cast<u8>((g * 255u + 31u) / 63u);
            rgba8[x * 4 + 2] = static_cast<u8>((b * 255u + 15u) / 31u);
            rgba8[x * 4 + 3] = 255;
        }
        return;
    case 4:  // RGBA4: rrrrggggbbbbaaaa
        for (u32 x = 0; x < width; ++x) {
            u16 value = 0;
            std::memcpy(&value, src + x * 2, 2);
            const u16 r = static_cast<u16>((value >> 12) & 0xF);
            const u16 g = static_cast<u16>((value >> 8) & 0xF);
            const u16 b = static_cast<u16>((value >> 4) & 0xF);
            const u16 a = static_cast<u16>(value & 0xF);
            rgba8[x * 4 + 0] = static_cast<u8>(r * 17u);
            rgba8[x * 4 + 1] = static_cast<u8>(g * 17u);
            rgba8[x * 4 + 2] = static_cast<u8>(b * 17u);
            rgba8[x * 4 + 3] = static_cast<u8>(a * 17u);
        }
        return;
    default:  // RGBA8
        std::memcpy(rgba8, src, static_cast<std::size_t>(width) * 4);
        return;
    }
}

// The fog table's 128 entries (two half-precision words each) as a key: the same table gives the same key, so the
// rasterizer uploads it once.
u32 FogLutKey(const std::array<u32, 128>& lut) {
    u32 hash = 2166136261u;
    for (const u32 entry : lut) {
        hash = (hash ^ entry) * 16777619u;
    }
    return hash != 0 ? hash : 1;  // 0 is "not uploaded"
}

void Drawer::Draw(const pica::DrawCall& draw) {
    const pica::PicaState& state = *draw.state;
    const auto& regs = state.regs;
    ++stats_.draws;
    // The primitive assembler applies the draw's register side effects first, whatever happens to the draw.
    assembler_.ApplyRegisterWrites(state);
    const u32 operation_mode = pica::BitsOf(regs[kRegOutputMerger], 0, 2);
    if (operation_mode == kShadowMode) {
        // A shadow draw renders into no attachment (the shadow map is the Vulkan path's): nothing to show here.
        ++stats_.shadow;
        return;
    }
    if (operation_mode == kGasDensityMode) {
        // The fog's density pass writes a density map, not a picture.
        ++stats_.gas;
        return;
    }
    if (!cpuvertex::NeedsCpuPath(draw, assembler_.IsEmpty())) {
        // A draw Azahar accelerates on the device's vertex stage: the CPU vertex unit shades it here too, so it is drawn
        // (correctly, if not as quickly as the Vulkan path's translated shader would).
        ++stats_.accelerated;
    }
    if (runner_.Run(host_, draw) != cpuvertex::RunStatus::Ok) {
        ++stats_.dropped;
        return;
    }
    batch_.Clear();
    assembler_.SubmitStream(runner_.output(), [this](const cpuvertex::OutputVertex& v0,
                                                      const cpuvertex::OutputVertex& v1,
                                                      const cpuvertex::OutputVertex& v2) {
        batch_.AddTriangle(v0, v1, v2);
    });
    if (batch_.Empty()) {
        return;
    }

    // The framebuffer the draw goes into, and the viewport and scissor of its registers.
    const u32 size_word = regs[kRegFramebufferSize];
    const u32 width = pica::BitsOf(size_word, 0, 11);
    const u32 height = pica::BitsOf(size_word, 12, 10) + 1;
    const bool flipped = pica::BitsOf(size_word, 24, 1) == 0;
    if (width == 0 || height == 0) {
        ++stats_.dropped;
        return;
    }
    const u32 colour_address = pica::BitsOf(regs[kRegColourAddress], 0, 28) * 8;
    target_ = render::Target{width, height, host_.VirtualToPhysical(colour_address).value_or(colour_address)};

    render::DrawState draw_state;
    MakeRectangles(regs, height, flipped, &draw_state);
    MakeFixedFunction(regs, flipped, &draw_state);
    MakeCombiner(regs, &draw_state.fragment);
    const u32 fog_word = regs[kRegFogModeAndMasks];
    draw_state.fragment.fog_mode = pica::BitsOf(fog_word, 0, 3);
    draw_state.fragment.fog_flip = pica::BitsOf(fog_word, 16, 1) != 0;
    const u32 fog_colour = regs[kRegFogColour];
    for (u32 channel = 0; channel < 3; ++channel) {
        draw_state.fragment.fog_colour[channel] = static_cast<float>(pica::BitsOf(fog_colour, 8 * channel, 8)) / 255.0f;
    }
    // (the table is the state's, not a register's: the game writes it through the LUT data port)
    draw_state.fragment.fog_lut = state.fog.lut.data();
    draw_state.fragment.fog_lut_key = FogLutKey(state.fog.lut);

    // The draw's textures, unit by unit, decoded by the texture cache: each image is uploaded once per generation.
    texture::TexturesOfDraw(regs.data(), texture_refs_, nullptr);
    draw_state.texture_count = 0;
    for (u32 unit = 0; unit < 3; ++unit) {
        const auto ref = std::find_if(texture_refs_.begin(), texture_refs_.end(),
                                      [unit](const texture::TextureRef& r) { return r.unit == unit; });
        if (ref == texture_refs_.end()) {
            continue;
        }
        const texture::CachedTexture* cached =
            textures_.Get(texture::CacheKey{ref->paddr, ref->format, ref->width, ref->height});
        if (cached == nullptr || cached->rgba8.empty()) {
            continue;
        }
        render::Texture& texture = draw_state.textures[unit];
        texture.rgba8 = cached->rgba8.data();
        texture.width = ref->width;
        texture.height = ref->height;
        // (the generation changes when the memory was decoded again: a new image, not the one uploaded for the old)
        texture.key = (static_cast<u64>(cached->generation) << 32) ^ static_cast<u64>(ref->paddr);
        const u32 params = regs[kRegTextureUnit[unit] + kTextureParams];
        texture.wrap_t = pica::BitsOf(params, 8, 3);
        texture.wrap_s = pica::BitsOf(params, 12, 3);
        // Unit 2 reads texture coordinate 1 instead of 2 when the main config says so (texture2_use_coord1).
        texture.coordinate = unit == 2 && pica::BitsOf(regs[kRegMainConfig], 13, 1) != 0 ? 1 : unit;
        draw_state.texture_count = static_cast<int>(unit) + 1;
    }

    // The pass's first draw loads the buffer's existing picture from guest memory (the game's fills, and what earlier
    // passes drew): only then is it needed, so the read is skipped while the pass goes on.
    render::TargetContents contents;
    if (!render::IsPassStarted(target_)) {
        const u32 format = pica::BitsOf(regs[kRegColourFormat], 16, 3);
        const u32 bpp = PicaBytesPerPixel(format);
        const u32 stride = width * bpp;
        bool read = stride != 0;
        seed_.assign(static_cast<std::size_t>(width) * height * 4, 0);
        for (u32 y = 0; read && y < height; ++y) {
            const u8* row = host_.PhysicalPointer(target_.address + y * stride, stride);
            if (row == nullptr) {
                read = false;
                break;
            }
            DecodeColourRow(row, width, format, seed_.data() + static_cast<std::size_t>(y) * width * 4);
        }
        if (read) {
            contents.rgba8 = seed_.data();
            contents.width = width;
            contents.height = height;
        }
    }

    std::string error;
    if (!render::Begin(target_, &contents, &error)) {
        ++stats_.failed;
        if (!told_failure_) {
            told_failure_ = true;
            MH4N_LOG(Error, kLog, "the rasterizer has no target: {}", error);
        }
        return;
    }
    ++stats_.cpu_draws;
    stats_.triangles += batch_.TriangleCount();
    if (!render::DrawTriangles(batch_.Data(), static_cast<u32>(batch_.VertexCount()), draw_state, &error)) {
        ++stats_.failed;
        if (!told_failure_) {
            told_failure_ = true;
            MH4N_LOG(Error, kLog, "a draw did not go through: {}", error);
        }
    }
}

bool Drawer::WriteColourToGuest(u32 address, u32 stride, u32 height, u32 format) {
    u32 width = 0;
    u32 rows = 0;
    std::string error;
    if (!render::ReadTarget(address, picture_, &width, &rows, &error)) {
        return false;
    }
    rows = std::min(rows, height);
    const u32 bpp = PicaBytesPerPixel(format);
    if (bpp == 0 || width == 0 || rows == 0 || stride < width * bpp) {
        return false;
    }
    for (u32 y = 0; y < rows; ++y) {
        u8* row = host_.PhysicalPointer(address + y * stride, width * bpp);
        if (row == nullptr) {
            return false;
        }
        EncodeColourRow(picture_.data() + static_cast<std::size_t>(y) * width * 4, width, format, row);
    }
    ++stats_.readbacks;
    return true;
}

}  // namespace mh4n::gpu::gles
