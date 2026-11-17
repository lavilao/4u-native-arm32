// The PICA's draws on the GLES rasterizer (gles_draw.h).
#include "gpu/gles/gles_draw.h"

#include <algorithm>
#include <cstring>

#include "gpu/pica/regs.h"
#include "mh4n/log.h"

namespace mh4n::gpu::gles {

namespace {

constexpr const char* kLog = "GlesGpu";

// RegsInternal: FramebufferRegs at 0x100, its framebuffer block at 0x110; RasterizerRegs at 0x40 (as native_gpu.cpp).
constexpr u32 kRegViewportSizeX = 0x41;
constexpr u32 kRegViewportSizeY = 0x43;
constexpr u32 kRegScissorXY1 = 0x66;
constexpr u32 kRegScissorXY2 = 0x67;
constexpr u32 kRegViewportCorner = 0x68;
constexpr u32 kRegColorFormat = 0x117;
constexpr u32 kRegFramebufferSize = 0x11E;

// Azahar's Common::Rectangle: top >= bottom (y up).
struct Rect4 {
    s32 left = 0, top = 0, right = 0, bottom = 0;
};

s32 Signed10(u32 v) {
    return static_cast<s32>(v << 22) >> 22;
}

// MakeGeometry of native_gpu.cpp, read into GL's pixels: the framebuffer's y axis points down there, so a rectangle
// whose top is `top` starts at `height - top`.
render::DrawState MakeDrawState(const std::array<u32, pica::kNumRegs>& regs, u32 height, bool flipped) {
    render::DrawState out;
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
    out.viewport_x = static_cast<float>(draw.left);
    out.viewport_y = static_cast<float>(h - draw.top);
    out.viewport_width = static_cast<float>(std::abs(draw.right - draw.left));
    out.viewport_height = static_cast<float>(std::abs(draw.top - draw.bottom));

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
    out.scissor_x = x0;
    out.scissor_y = h - y1;
    out.scissor_width = static_cast<u32>(std::max(x1 - x0, 0));
    out.scissor_height = static_cast<u32>(std::max(y1 - y0, 0));
    // The depth test is always on for now (the output merger's registers are not read yet); depth writes with it.
    out.depth_test = true;
    out.depth_write = true;
    return out;
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

void Drawer::Draw(const pica::DrawCall& draw) {
    const pica::PicaState& state = *draw.state;
    const auto& regs = state.regs;
    ++stats_.draws;
    // The primitive assembler applies the draw's register side effects first, whatever happens to the draw.
    assembler_.ApplyRegisterWrites(state);
    if (!cpuvertex::NeedsCpuPath(draw, assembler_.IsEmpty())) {
        // An accelerated draw: Azahar draws it with the translated vertex stage, which this path does not have yet.
        runner_.NoteDraw(draw);
        ++stats_.accelerated;
        return;
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
    target_ = render::Target{width, height};
    const render::DrawState draw_state = MakeDrawState(regs, height, flipped);

    // The draw's first texture, decoded by the texture cache: its pixels are uploaded once per generation.
    texture::TexturesOfDraw(regs.data(), texture_refs_, nullptr);
    render::Texture texture;
    const render::Texture* texture_ptr = nullptr;
    if (!texture_refs_.empty()) {
        const texture::TextureRef& ref = texture_refs_.front();
        const texture::CachedTexture* cached =
            textures_.Get(texture::CacheKey{ref.paddr, ref.format, ref.width, ref.height});
        if (cached != nullptr && !cached->rgba8.empty()) {
            texture.rgba8 = cached->rgba8.data();
            texture.width = ref.width;
            texture.height = ref.height;
            // (the generation changes when the memory was decoded again: a new image, not the one uploaded for the old)
            texture.key = (static_cast<u64>(cached->generation) << 32) ^ static_cast<u64>(ref.paddr);
            texture_ptr = &texture;
        }
    }

    std::string error;
    if (!render::Begin(target_, &error)) {
        ++stats_.failed;
        if (!told_failure_) {
            told_failure_ = true;
            MH4N_LOG(Error, kLog, "the rasterizer has no target: {}", error);
        }
        return;
    }
    ++stats_.cpu_draws;
    stats_.triangles += batch_.TriangleCount();
    if (!render::DrawTriangles(batch_.Data(), static_cast<u32>(batch_.VertexCount()), draw_state, texture_ptr, &error)) {
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
    if (!render::ReadTarget(picture_, &width, &rows, &error)) {
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
