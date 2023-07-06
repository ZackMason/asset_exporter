#pragma once

// note(zack): this file is the sole interface between the app and the platform layer

#define WIN32_MEAN_AND_LEAN
#define NOMINMAX

#if defined(RAND_GETPID)

#elif defined(_WIN64) || defined(_WIN32)
    #include <process.h>
    #define RAND_GETPID _getpid()
    // idk if needed
    #undef min
    #undef max
    #undef near
    #undef far
#elif defined(__unix__) || defined(__unix) \
      || (defined(__APPLE__) && defined(__MACH__))
    #include <unistd.h>
    #define RAND_GETPID getpid()
#else
    #define RAND_GETPID 0
#endif


#include <cstdint>
#include <cassert>
#include <cstring>
#include <stack>
#include <span>
#include <array>
#include <optional>
#include <vector>

#include <string>
#include <string_view>



#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <fmt/color.h>



struct aiNodeAnim;

///////////////////////////////////////////////////////////////////////////////
// Type Defs
///////////////////////////////////////////////////////////////////////////////
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

// #include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

using v2f = glm::vec2;
using v3f = glm::vec3;
using v4f = glm::vec4;

using v2i = glm::ivec2;
using v3i = glm::ivec3;
using v4i = glm::ivec4;

using quat = glm::quat;
using m22 = glm::mat2x2;
using m33 = glm::mat3x3;
using m34 = glm::mat3x4;
using m43 = glm::mat4x3;
using m44 = glm::mat4x4;
///////////////////////////////////////////////////////////////////////////////

#define fmt_str(...) (fmt::format(__VA_ARGS__))
#define println(...) do { fmt::print(__VA_ARGS__); } while(0)
#define gen_info(cat, str, ...) do { fmt::print(fg(fmt::color::white) | fmt::emphasis::bold, fmt_str("[info][{}]: {}\n", cat, str), __VA_ARGS__); } while(0)
#define gen_warn(cat, str, ...) do { fmt::print(stderr, fg(fmt::color::yellow) | fmt::emphasis::bold, fmt_str("[warn][{}]: {}\n", cat, str), __VA_ARGS__); } while(0)
#define gen_error(cat, str, ...) do { fmt::print(stderr, fg(fmt::color::red) | fmt::emphasis::bold, fmt_str("[error][{}]: {}\n", cat, str), __VA_ARGS__); } while(0)

#define global_variable static
#define local_persist static

#define no_mangle extern "C"
#define export_dll __declspec(dllexport)
#define export_fn(rt_type) no_mangle rt_type export_dll __cdecl

#define array_count(arr) (sizeof((arr)) / (sizeof((arr)[0])))
#define kilobytes(x) (x*1024ull)
#define megabytes(x) (kilobytes(x)*1024ull)
#define gigabytes(x) (megabytes(x)*1024ull)
#define terabytes(x) (gigabytes(x)*1024ull)
#define align16(val) ((val + 15) & ~15)
#define align4(val) ((val + 3) & ~3)

inline u32
safe_truncate_u64(u64 value) {
    assert(value <= 0xFFFF'FFFF);
    const u32 result = static_cast<u32>(value);
    return result;
}

constexpr int 
BIT(int x) {
	return 1 << x;
}


// nodes

#define node_next(node) (node = node->next)
#define node_push(node, list) do { node->next = list; list = node; } while(0)
#define node_pop(node, list) do { node = list; list = list->next; } while(0)

#define node_for(type, node, itr) for(type* itr = (node); (itr); node_next(itr))

template <typename T>
struct node_t {
    union {
        T* left;
        T* prev;
    };
    union {
        T* right;
        T* next;
    };
    node_t() {
        right = left = nullptr;
    }
};

// allocators

struct arena_t {
    u8* start{0};
    size_t top{0};
    size_t size{0};
};

struct frame_arena_t {
    arena_t arena[2];
    u32     active{0};
};

using temp_arena_t = arena_t;

inline size_t
arena_get_mark(arena_t* arena) {
    return arena->top;
}

inline void
arena_set_mark(arena_t* arena, size_t m) {
    arena->top = m;
}

inline void
arena_clear(arena_t* arena) {
    arena->top = 0;
}

inline u8*
arena_get_top(arena_t* arena) {
    return arena->start + arena->top;
}

#define arena_display_info(arena, name) \
    (fmt_str("{}: {} {} / {} {} - {:.2f}%", \
    (name), \
    (arena)->top > gigabytes(1) ? (arena)->top/gigabytes(1) : (arena)->top > megabytes(1) ? (arena)->top/megabytes(1) : (arena)->top,\
    (arena)->top > gigabytes(1) ? "Gb" : (arena)->top > megabytes(1) ? "Mb" : "B",\
    (arena)->size > gigabytes(1) ? (arena)->size/gigabytes(1) : (arena)->size > megabytes(1) ? (arena)->size/megabytes(1) : (arena)->size,\
    (arena)->size > gigabytes(1) ? "Gb" : (arena)->size > megabytes(1) ? "Mb" : "B",\
    (f32((arena)->top) / f32((arena)->size) * 100.0f)))

inline u8*
arena_alloc(arena_t* arena, size_t bytes) {
    u8* p_start = arena->start + arena->top;
    arena->top += bytes;

    assert(arena->top <= arena->size && "Arena overflow");

    return p_start;
}

template <typename T>
inline T*
arena_alloc_ctor(arena_t* arena, size_t count = 1) {
    T* t = reinterpret_cast<T*>(arena_alloc(arena, sizeof(T) * count));
    for (size_t i = 0; i < count; i++) {
        new (t + i) T();
    }
    return t;
}

template <typename T, typename ... Args>
inline T*
arena_alloc_ctor(arena_t* arena, size_t count, Args&& ... args) {
    T* t = reinterpret_cast<T*>(arena_alloc(arena, sizeof(T) * count));
    for (size_t i = 0; i < count; i++) {
        new (t + i) T(std::forward<Args>(args)...);
    }
    return t;
}

template <typename T, typename ... Args>
T* arena_emplace(arena_t* arena, size_t count, Args&& ... args) {
    T* t = reinterpret_cast<T*>(arena_alloc(arena, sizeof(T) * count));
    for (size_t i = 0; i < count; i++) {
        new (t + i) T(std::forward<Args>(args)...);
    }
    return t;
}

arena_t 
arena_sub_arena(arena_t* arena, size_t size) {
    arena_t res;
    res.start = arena_alloc(arena, size);
    res.size = size;
    res.top = 0;
    return res;
}

inline frame_arena_t
arena_create_frame_arena(
    arena_t* arena, 
    size_t bytes
) {
    frame_arena_t frame;
    frame.arena[0].start = arena_alloc(arena, bytes);
    frame.arena[1].start = arena_alloc(arena, bytes);
    frame.arena[0].size = bytes;
    frame.arena[1].size = bytes;
    return frame;
}

namespace utl {

template <typename T>
struct pool_t : public arena_t {
    size_t count{0};
    size_t capacity{0};

    pool_t() = default;

    explicit pool_t(T* data, size_t _cap) 
        : capacity{_cap}
    {
        start = (u8*)data;
        size = _cap * sizeof(T);
    }

    pool_t(arena_t* arena, size_t _cap)
        : capacity{_cap}
    {
        auto sub_arena = arena_sub_arena(arena, _cap*sizeof(T));
        start = sub_arena.start;
        size = sub_arena.size;
    }

    T& operator[](size_t id) {
        return data()[id];
    }

    T& back() {
        return operator[](count-1);
    }

    T& front() {
        return operator[](0);
    }

    T* data() {
        return (T*)start;
    }

    T* create(size_t p_count) {
        T* o = allocate(p_count);
        for (size_t i = 0; i < p_count; i++) {
            new (o+i) T();
        }
        return o;
    }

    T* allocate(size_t p_count) {
        count += p_count;
        return (T*)arena_alloc(this, sizeof(T) * p_count);
    }

    void free(T* ptr) {
        ptr->~T();
        std::swap(*ptr, back());
        count--;
    }

    // note(zack): calls T dtor
    void clear() {
        if constexpr (std::is_trivially_destructible_v<T>) {
            for (size_t i = 0; i < count; i++) {
                (data() + i)->~T();
            }
        }
        arena_clear(this);
        count = 0;
    }    
};

}; // namespace utl


// string

struct string_t {
    union {
        char* data;
        const char* c_data;
    };
    size_t size;
    bool owns = false;

    string_t()
        : data{nullptr}, size{0}
    {

    }

    const char* c_str() const {
        return c_data;
    }

    void view(const char* str) {
        size = std::strlen(str) + 1;
        c_data = str;
        owns = false;
    }
    
    void own(arena_t* arena, const char* str) {
        size = std::strlen(str) + 1;
        data = (char*)arena_alloc(arena, size);
        std::memcpy(data, str, size);
        owns = true;
    }
};

struct string_node_t : node_t<string_node_t> {
    string_t string;
};

using string_id_t = u64;
using sid_t = string_id_t;


// https://stackoverflow.com/questions/48896142/is-it-possible-to-get-hash-values-as-compile-time-constants
template <typename Str>
static constexpr string_id_t sid(const Str& toHash) {
    static_assert(sizeof(string_id_t) == 8, "string hash not u64");
	string_id_t result = 0xcbf29ce484222325; // FNV offset basis

	for (char c : toHash) {
		result ^= c;
		result *= 1099511628211; // FNV prime
	}

	return result;
}

template <size_t N>
static constexpr string_id_t sid(char const (&toHash)[N]) {
	return sid(std::string_view(toHash));
}

constexpr string_id_t operator"" _sid(const char* str, std::size_t size) {
    return sid(std::string_view(str));
}

namespace math {
    template <typename T>
    struct hit_result_t {
        bool intersect{false};
        f32 distance{0.0f};
        v3f position{0.f, 0.f, 0.f};
        T& object;

        operator bool() const {
            return intersect;
        }
    };

    struct ray_t {
        v3f origin;
        v3f direction;
    };
    
template <typename T = v3f>
struct aabb_t {
    T min{std::numeric_limits<float>::max()};
    T max{-std::numeric_limits<float>::max()};

    T size() const {
        return max - min;
    }

    T center() const {
        return (max+min)/2.0f;
    }

    void expand(const aabb_t<T>& o) {
        expand(o.min);
        expand(o.max);
    }

    void expand(const T& p) {
        min = glm::min(p, min);
        max = glm::max(p, max);
    }

    bool contains(const auto& p) const {
        //static_assert(std::is_same<T, v3f>::value);
        return min <= p && max <= p;
        // return 
        //     min.x <= p.x &&
        //     min.y <= p.y &&
        //     min.z <= p.z &&
        //     p.x <= max.x &&
        //     p.y <= max.y &&
        //     p.z <= max.z;
    }

    // bool contains(const v2f& p) const {
    //     //static_assert(std::is_same<T, v3f>::value);
    
    //     return 
    //         min.x <= p.x &&
    //         min.y <= p.y &&
    //         p.x <= max.x &&
    //         p.y <= max.y;
    // }

    // bool contains(const f32 p) const {
    //     static_assert(std::is_same<T, f32>::value);
    //     return min <= p && p <= max;
    // }
};

struct transform_t {
	transform_t(const m44& mat = m44{1.0f}) : basis(mat), origin(mat[3]) {};
	transform_t(const v3f& position, const v3f& scale = {1,1,1}, const v3f& rotation = {0,0,0})
	    : basis(m33(1.0f)
    ) {
		origin = (position);
		set_rotation(rotation);
		set_scale(scale);
	}
    
	m44 to_matrix() const {
        m44 res = basis;
        res[3] = v4f(origin,1.0f);
        return res;
    }
    
	operator m44() const {
		return to_matrix();
	}
    
	void translate(const v3f& delta) {
        origin += delta;
    }
	void scale(const v3f& delta) {
        basis = glm::scale(m44{basis}, delta);
    }
	void rotate(const v3f& delta) {
        m44 rot = m44(1.0f);
        rot = glm::rotate(rot, delta.z, { 0,0,1 });
        rot = glm::rotate(rot, delta.y, { 0,1,0 });
        rot = glm::rotate(rot, delta.x, { 1,0,0 });
        basis = (basis) * m33(rot);
    }
	void rotate_quat(const glm::quat& delta) {
        set_rotation(get_orientation() * delta);
    }
    
    glm::quat get_orientation() const {
        return glm::quat_cast(basis);
    }

	transform_t inverse() const {
        transform_t transform;
        transform.basis = glm::transpose(basis);
        transform.origin = basis * -origin;
        return transform;
    }

    
	void look_at(const v3f& target, const v3f& up = { 0,1,0 }) {
        auto mat = glm::lookAt(origin, target, up);
        basis = mat;
    }
    void set_scale(const v3f& scale) {
        for (int i = 0; i < 3; i++) {
            basis[i] = glm::normalize(basis[i]) * scale[i];
        }
    }
	void set_rotation(const v3f& rotation) {
        basis = glm::eulerAngleYXZ(rotation.y, rotation.x, rotation.z);
    }
	void set_rotation(const glm::quat& quat) {
        basis = glm::toMat3(glm::normalize(quat));
    }

    void normalize() {
        for (decltype(basis)::length_type i = 0; i < 3; i++) {
            basis[i] = glm::normalize(basis[i]);
        }
    }

	// in radians
	v3f get_euler_rotation() const {
        return glm::eulerAngles(get_orientation());
    }

	void affine_invert() {
		basis = glm::inverse(basis);
		origin = basis * -origin;
	}

	v3f inv_xform(const v3f& vector) const
	{
		const v3f v = vector - origin;
		return glm::transpose(basis) * v;
	}

	v3f xform(const v3f& vector)const
	{
		return v3f(basis * vector) + origin;
	}
    
	ray_t xform(const ray_t& ray) const { 
		return ray_t{
			xform(ray.origin),
			glm::normalize(basis * ray.direction)
		};
	}

	aabb_t<v3f> xform_aabb(const aabb_t<v3f>& box) const {
		aabb_t<v3f> t_box;
		t_box.min = t_box.max = origin;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
                const auto a = basis[j][i] * box.min[j];
                const auto b = basis[j][i] * box.max[j];
                t_box.min[i] += a < b ? a : b;
                t_box.max[i] += a < b ? b : a;
            }
        }

		return t_box;
	}
    
	m33 basis = m33(1.0f);
	v3f origin = v3f(0, 0, 0);
};


}; // namespace math


namespace gfx {

// note(zack): this is in core because it will probably be loaded 
// in a work thread in platform layer, so that we dont have to worry
// about halting the thread when we reload the dll


using color4 = v4f;
using color3 = v3f;
using color32 = u32;
struct font_t;

struct material_info_t {
    using stack_string_t = char[256];
    stack_string_t textures[8]{};
    stack_string_t properties[32]{};
    u8 transparency{0};
};

struct material_t {
    v4f albedo{};

    f32 ao{};
    f32 emission{};
    f32 metallic{};
    f32 roughness{};

    u32 flags{};     // for material effects
    u32 opt_flags{}; // for performance
    u32 padding[2]{};

    u64 textures[8]{};
};

namespace gui {
    struct vertex_t;
};

inline void
font_render(
    arena_t* arena,
    font_t* font,
    std::string_view text,
    v2f position,
    v2f screen_size,
    utl::pool_t<gui::vertex_t>* vertices,
    utl::pool_t<u32>* indices,
    color32 text_color);


namespace color {
    constexpr color32 to_color32(const color4& c) {
        return 
            (u8(std::min(u16(c.x * 255.0f), 255ui16)) << 0 ) |
            (u8(std::min(u16(c.y * 255.0f), 255ui16)) << 8 ) |
            (u8(std::min(u16(c.z * 255.0f), 255ui16)) << 16) |
            (u8(std::min(u16(c.w * 255.0f), 255ui16)) << 24) ;
    }

    constexpr color32 to_color32(const color3& c) {
        return to_color32(color4(c, 1.0f));
    }

    constexpr color4 to_color4(const color32 c) {
        return color4{
            ((c&0xff)       >> 0),
            ((c&0xff00)     >> 8),
            ((c&0xff0000)   >> 16),
            ((c&0xff000000) >> 24)            
        } / 255.0f;
    }

    constexpr bool is_hex_digit(char c) {
        return ('0' <= c && c <= '9') || ('a' <= c && c <= 'f') || ('A' <= c && c <= 'F');
    }

    constexpr u32 hex_value(char c) {
        constexpr std::array<std::pair<char, u32>, 22> lut {{
            {'0', 0u}, {'1', 1u}, {'2', 2u}, {'3', 3u}, {'4', 4u},
            {'5', 5u}, {'6', 6u}, {'7', 7u}, {'8', 8u}, {'9', 9u},
            {'a', 10u},{'b', 11u},{'c', 12u},{'d', 13u},{'e', 14u},
            {'f', 15u},{'A', 10u},{'B', 11u},{'C', 12u},{'D', 13u},
            {'E', 14u},{'F', 15u},
        }};
        for (const auto& [k, v]: lut) { 
            if (k == c) return v;
        }

        return 0;
    }

    template <typename Str>
    constexpr color4 str_to_rgba(const Str& str) {
        assert(str.size() == 9 && "invalid rgba size");

        color4 res;
        res.x = f32(hex_value(str[1]) * 16u + hex_value(str[2])) / 255.0f;
        res.y = f32(hex_value(str[3]) * 16u + hex_value(str[4])) / 255.0f;
        res.z = f32(hex_value(str[5]) * 16u + hex_value(str[6])) / 255.0f;
        res.w = f32(hex_value(str[7]) * 16u + hex_value(str[8])) / 255.0f;

        res = glm::min(res, color4(1.0f));

        return res;
    }

    constexpr color32 operator"" _rgba(const char* str, std::size_t size) {
        auto c = str_to_rgba(std::string_view(str, size));
        return to_color32(c);
    }

    constexpr color32 operator"" _abgr(const char* str, std::size_t size) {
        auto c = str_to_rgba(std::string_view(str, size));
        u32 res = 0u;
        res = 
            (u8(c.x * 255.0f) << 24) |
            (u8(c.y * 255.0f) << 16) |
            (u8(c.z * 255.0f) << 8)  |
            (u8(c.w * 255.0f) << 0)  ;

        return res; 
    }

    constexpr color4 operator"" _c4(const char* str, std::size_t size) {
        return str_to_rgba(std::string_view(str, size));
    }

    constexpr color3 operator"" _c3(const char* str, std::size_t size) {
        return color3{str_to_rgba(std::string_view(str, size))};
    }

    namespace rgba {
        static auto clear = "#00000000"_rgba;
        static auto black = "#000000ff"_rgba;
        static auto dark_gray = "#111111ff"_rgba;
        static auto white = "#ffffffff"_rgba;
        static auto cream = "#fafafaff"_rgba;
        static auto red   = "#ff0000ff"_rgba;
        static auto green = "#00ff00ff"_rgba;
        static auto blue  = "#0000ffff"_rgba;
        static auto yellow= "#ffff00ff"_rgba;
        static auto purple= "#fa11faff"_rgba;
        static auto sand  = "#C2B280ff"_rgba;
    };

    namespace abgr {
        static auto clear = "#00000000"_abgr;
        static auto black = "#000000ff"_abgr;
        static auto white = "#ffffffff"_abgr;
        static auto red   = "#ff0000ff"_abgr;
        static auto green = "#00ff00ff"_abgr;
        static auto blue  = "#0000ffff"_abgr;
        static auto yellow= "#ffff00ff"_abgr;
        static auto sand  = "#C2B280ff"_abgr;
    };

    namespace v4 {
        static auto clear = "#00000000"_c4;
        static auto black = "#000000ff"_c4;
        static auto white = "#ffffffff"_c4;
        static auto ray_white   = "#f1f1f1ff"_c4;
        static auto light_gray  = "#d3d3d3ff"_c4;
        static auto dark_gray   = "#2a2a2aff"_c4;
        static auto red   = "#ff0000ff"_c4;
        static auto green = "#00ff00ff"_c4;
        static auto blue  = "#0000ffff"_c4;
        static auto purple= "#ff00ffff"_c4;
        static auto cyan  = "#00ffffff"_c4;
        static auto yellow= "#ffff00ff"_c4;
        static auto sand  = "#C2B280ff"_c4;
    };

    namespace v3 {
        static auto clear = "#00000000"_c3;
        static auto black = "#000000ff"_c3;
        static auto white = "#ffffffff"_c3;
        static auto gray  = "#333333ff"_c3;
        static auto red   = "#ff0000ff"_c3;
        static auto green = "#00ff00ff"_c3;
        static auto cyan  = "#00ffffff"_c3;
        static auto blue  = "#0000ffff"_c3;
        static auto purple= "#ff00ffff"_c3;
        static auto yellow= "#ffff00ff"_c3;
        static auto sand  = "#C2B280ff"_c3;
    };

}; // namespace color


namespace gui {
    struct vertex_t {
        v2f pos;
        v2f tex;
        u32 img;
        u32 col;
    };

    struct ctx_t {
        utl::pool_t<vertex_t>* vertices;
        utl::pool_t<u32>* indices;
        gfx::font_t* font;
        v2f screen_size{};
    };

    enum eTypeFlag {
        eTypeFlag_Theme,
        eTypeFlag_Panel,
        eTypeFlag_Text,

        eTypeFlag_None,
    };

    struct theme_t {
        color32 fg_color{};
        color32 bg_color{};
        color32 text_color{color::rgba::cream};
        color32 border_color{};
        u32     shadow_distance{};
        color32 shadow_color{};
    };

    struct text_t;
    struct image_t;
    struct button_t;

    struct panel_t : public math::aabb_t<v2f> {
        eTypeFlag flag{eTypeFlag_Panel};
        panel_t* next{nullptr};

        text_t* text_widgets{nullptr};
        button_t* button_widgets{nullptr};

        sid_t name;

        theme_t theme;
    };

    struct button_t : public math::aabb_t<v2f> {
        void* user_data{0};
        
        using callback_fn = void(*)(void*);
        callback_fn on_press{0};

        theme_t theme;
    };

    struct text_t {
        eTypeFlag type{eTypeFlag_Text};
        text_t* next{nullptr};

        string_t text{};
        v2f offset{};
        u32 font_size{};

        theme_t theme;
    };
    
    inline void
    ctx_clear(
        ctx_t* ctx
    ) {
        ctx->vertices->clear();
        ctx->indices->clear();
    }

    inline void 
    text_render(
        ctx_t* ctx,
        panel_t* parent,
        text_t* text
    ) {
        font_render(0, ctx->font, 
            text->text.c_str(), 
            parent->min + text->offset,
            ctx->screen_size,
            ctx->vertices,
            ctx->indices,
            text->theme.text_color
        );

        if (text->theme.shadow_distance) {
            font_render(0, ctx->font, 
                text->text.c_str(), 
                parent->min + text->offset + v2f{(f32)text->theme.shadow_distance},
                ctx->screen_size,
                ctx->vertices,
                ctx->indices,
                text->theme.shadow_color
            );
        }

        if (text->next) {
            text_render(ctx, parent, text->next);
        }
    }

    inline void
    panel_render(
        ctx_t* ctx,
        panel_t* panel
    ) {
        if (panel->next) { panel_render(ctx, panel->next); }

        if (panel->text_widgets) {
            text_render(ctx, panel, panel->text_widgets);
        }

        const u32 v_start = safe_truncate_u64(ctx->vertices->count);

        const v2f p0 = panel->min;
        const v2f p1 = v2f{panel->min.x, panel->max.y};
        const v2f p2 = v2f{panel->max.x, panel->min.y};
        const v2f p3 = panel->max;

        vertex_t* v = ctx->vertices->allocate(4);
        u32* i = ctx->indices->allocate(6);
        v[0] = vertex_t { .pos = p0 / ctx->screen_size, .tex = v2f{0.0f, 1.0f}, .img = ~(0ui32), .col = panel->theme.bg_color};
        v[1] = vertex_t { .pos = p1 / ctx->screen_size, .tex = v2f{0.0f, 0.0f}, .img = ~(0ui32), .col = panel->theme.bg_color};
        v[2] = vertex_t { .pos = p2 / ctx->screen_size, .tex = v2f{1.0f, 0.0f}, .img = ~(0ui32), .col = panel->theme.bg_color};
        v[3] = vertex_t { .pos = p3 / ctx->screen_size, .tex = v2f{1.0f, 1.0f}, .img = ~(0ui32), .col = panel->theme.bg_color};

        i[0] = v_start + 0;
        i[1] = v_start + 1;
        i[2] = v_start + 2;

        i[3] = v_start + 2;
        i[4] = v_start + 1;
        i[5] = v_start + 3;

    }

}; // namespace gui


struct vertex_t {
    v3f pos;
    v3f nrm;
    color3 col;
    v2f tex;
};

struct skinned_vertex_t {
    v3f pos;
    v3f nrm;
    v2f tex;
    // v3f tangent;
    // v3f bitangent;
    u32 bone_id[4];
    v4f weight{0.0f};
};

using index32_t = u32;

struct mesh_t {
    vertex_t*   vertices{nullptr};
    u32         vertex_count{0};
    index32_t*  indices{nullptr};
    u32         index_count{0};
};

struct mesh_view_t {
    u32 vertex_start{};
    u32 vertex_count{};
    u32 index_start{};
    u32 index_count{};
    u32 instance_start{};
    u32 instance_count{1};

    math::aabb_t<v3f> aabb{};
};

}; // namespace gfx

namespace utl {
    
namespace rng {

uint64_t fnv_hash_u64(uint64_t val) {
    uint64_t hash = 14695981039346656037ull;
    for (uint64_t i = 0; i < 8; i++) {
        hash = hash ^ ((val >> (8 * i)) & 0xff);
        hash = hash * 1099511628211ull;
    }
    return hash;
}

struct xoshiro256_random_t {
    uint64_t state[4];
    constexpr static uint64_t max = std::numeric_limits<uint64_t>::max();

    void randomize() {
        constexpr u64 comp_time_entropy = sid(__TIME__) ^ sid(__DATE__);
        const u64 this_entropy = utl::rng::fnv_hash_u64((std::uintptr_t)this);
        const u64 pid_entropy = utl::rng::fnv_hash_u64(RAND_GETPID);
        
        if constexpr (sizeof(time_t) >= 8) {
            state[3] = static_cast<uint64_t>(time(0)) ^ comp_time_entropy;
            state[2] = (static_cast<uint64_t>(time(0)) << 17) ^ this_entropy;
            state[1] = (static_cast<uint64_t>(time(0)) >> 21) ^ (static_cast<uint64_t>(~time(0)) << 13);
            state[0] = (static_cast<uint64_t>(~time(0)) << 23) ^ pid_entropy;
        }
        else {
            state[3] = static_cast<uint64_t>(time(0) | (~time(0) << 32));
            state[2] = static_cast<uint64_t>(time(0) | (~time(0) << 32));
            state[1] = static_cast<uint64_t>(time(0) | (~time(0) << 32));
            state[0] = static_cast<uint64_t>(time(0) | (~time(0) << 32));
        }

        // todo(zack): remove this, get better seed
        for (size_t i = 0; i < 5; i++) { rand(); }
    };

    uint64_t rand() {
        uint64_t *s = state;
        uint64_t const result = rol64(s[1] * 5, 7) * 9;
        uint64_t const t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rol64(s[3], 45);

        return result;
    }

    uint64_t rol64(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

struct xor64_random_t {
    uint64_t state;
    constexpr static uint64_t max = std::numeric_limits<uint64_t>::max();

    void randomize() {
        if constexpr (sizeof(time_t) >= 8) {
            constexpr u64 comp_time_entropy = sid(__TIME__) ^ sid(__DATE__);
            const u64 this_entropy = utl::rng::fnv_hash_u64((std::uintptr_t)this);
            const u64 pid_entropy = utl::rng::fnv_hash_u64(RAND_GETPID);
        
            state = static_cast<uint64_t>(time(0)) ^ comp_time_entropy ^ this_entropy ^ pid_entropy;
            state = utl::rng::fnv_hash_u64(state);
        }
        else {
            state = static_cast<uint64_t>(time(0) | (~time(0) << 32));
        }
    };

    uint64_t rand(){
        auto x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return state = x;
    }
};

struct xor32_random_t {
    uint32_t state;
    constexpr static uint32_t max = std::numeric_limits<uint32_t>::max();

    void randomize() {
        constexpr u64 comp_time_entropy = sid(__TIME__) ^ sid(__DATE__);
        const u64 this_entropy = utl::rng::fnv_hash_u64((std::uintptr_t)this);
        const u64 pid_entropy = utl::rng::fnv_hash_u64(RAND_GETPID);
    
        state = static_cast<uint32_t>(time(0)) ^ (u32)(comp_time_entropy ^ this_entropy ^ pid_entropy);
        state = (u32)utl::rng::fnv_hash_u64(state);
    };

    uint32_t rand(){
        auto x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return state = x;
    }
};

template <typename Generator>
struct random_t {
    Generator rng;

    random_t(const decltype(Generator::state)& x = {}) noexcept {
        seed(x);
    }

    void randomize() {
        rng.randomize();
    }

    void seed(const decltype(Generator::state)& x) {
        if constexpr (std::is_integral_v<decltype(Generator::state)>) {
            if (x == 0) {
                randomize();
            } else {
                rng.state = x;
            }
        } else {
            randomize();
        }
    }

    auto rand() {
        return rng.rand();
    }

    float randf() {
        return float(rng.rand()) / float(Generator::max);
    }

    float randn() {
        return randf() * 2.0f - 1.0f;
    }

    template <typename T>
    auto choice(const T& indexable) -> const typename T::value_type& {
        return indexable[rng.rand()%indexable.size()];
    }

    v3f aabb(const math::aabb_t<v3f>& box) {
        return v3f{
            (this->randf() * (box.max.x - box.min.x)) + box.min.x,
            (this->randf() * (box.max.y - box.min.y)) + box.min.y,
            (this->randf() * (box.max.z - box.min.z)) + box.min.z
        };
    }
};

struct random_s {
    inline static random_t<xor64_random_t> state{0};

    template <typename T>
    static auto choice(const T& indexable) -> const typename T::value_type& {
        return state.choice(indexable);
    }

    static void randomize() {
        state.randomize();
    }
    //random float
    static float randf() {
        return state.randf();
    }
    static uint64_t rand() {
        return state.rand();
    }
    // random normal
    static float randn() {
        return state.randn();
    }
    template <typename AABB>
    static auto aabb(const AABB& box) -> decltype(box.min) {
        return state.aabb(box);
    }
};

}; // namespace rng


// Generated with ChatGPT
// A serialization stream class
struct serializer_t {
    explicit serializer_t(std::vector<uint8_t>& data) : data_(data) {}
    virtual ~serializer_t() = default;

    // Serialization functions
    inline void serialize(bool value)      { data_.push_back(value ? 1 : 0); }
    inline void serialize(int8_t value)    { data_.push_back(value); }
    inline void serialize(uint8_t value)   { data_.push_back(value); }
    inline void serialize(int16_t value)   { data_.insert(data_.end(), (uint8_t*)&value, (uint8_t*)&value + sizeof(value)); }
    inline void serialize(uint16_t value)  { data_.insert(data_.end(), (uint8_t*)&value, (uint8_t*)&value + sizeof(value)); }
    inline void serialize(int32_t value)   { data_.insert(data_.end(), (uint8_t*)&value, (uint8_t*)&value + sizeof(value)); }
    inline void serialize(uint32_t value)  { data_.insert(data_.end(), (uint8_t*)&value, (uint8_t*)&value + sizeof(value)); }
    inline void serialize(int64_t value)   { data_.insert(data_.end(), (uint8_t*)&value, (uint8_t*)&value + sizeof(value)); }
    inline void serialize(uint64_t value)  { data_.insert(data_.end(), (uint8_t*)&value, (uint8_t*)&value + sizeof(value)); }
    inline void serialize(float value)     { data_.insert(data_.end(), (uint8_t*)&value, (uint8_t*)&value + sizeof(value)); }
    inline void serialize(double value)    { data_.insert(data_.end(), (uint8_t*)&value, (uint8_t*)&value + sizeof(value)); }
    inline void serialize(std::string_view value) {
        size_t length = value.length();
        serialize(length);
        data_.insert(data_.end(), value.begin(), value.end());
    }

    template <typename T>
    inline void serialize(const std::vector<T>& value) {
        size_t size = value.size();
        serialize(size);
        for (const T& element : value) {
            serialize(element);
        }
    }
    template <typename T>
    inline void serialize(std::span<T> value) {
        size_t size = value.size();
        serialize(size);
        for (T& element : value) {
            serialize(element);
        }
    }
    template <typename T>
    inline void serialize(std::span<T> value, void(*serialize_impl)(serializer_t&, const T&)) {
        size_t size = value.size();
        serialize(size);
        for (T& element : value) {
            serialize_impl(*this, element);
        }
    }

private:
    std::vector<uint8_t>& data_;
};

struct deserializer_t {
    deserializer_t(const std::vector<uint8_t>& data) : m_Data(data), m_Offset(0) {}

    bool deserialize_bool() { return m_Data[m_Offset++] != 0; }
    int8_t deserialize_i8() { return m_Data[m_Offset++]; }
    uint8_t deserialize_u8() { return m_Data[m_Offset++]; }
    int16_t deserialize_i16() { int16_t value = *(int16_t*)&m_Data[m_Offset]; m_Offset += sizeof(value); return value; }
    uint16_t deserialize_u16() { uint16_t value = *(uint16_t*)&m_Data[m_Offset]; m_Offset += sizeof(value); return value; }
    int32_t deserialize_i32() { int32_t value = *(int32_t*)&m_Data[m_Offset]; m_Offset += sizeof(value); return value; }
    uint32_t deserialize_u32() { uint32_t value = *(uint32_t*)&m_Data[m_Offset]; m_Offset += sizeof(value); return value; }
    int64_t deserialize_i64() { int64_t value = *(int64_t*)&m_Data[m_Offset]; m_Offset += sizeof(value); return value; }
    uint64_t deserialize_u64() { uint64_t value = *(uint64_t*)&m_Data[m_Offset]; m_Offset += sizeof(value); return value; }
    float deserialize_f32() { float value = *(float*)&m_Data[m_Offset]; m_Offset += sizeof(value); return value; }
    double deserialize_f64() { double value = *(double*)&m_Data[m_Offset]; m_Offset += sizeof(value); return value; }

    std::string deserialize_string() {
        size_t length = deserialize_u64();
        std::string value((char*)&m_Data[m_Offset], length);
        m_Offset += length;
        return value;
    }
    void deserialize_string_view(std::string_view s) {
        std::memcpy(const_cast<char*>(s.data()), (char*)&m_Data[m_Offset], s.size());
        m_Offset += s.size();
    }

    template <typename T>
    std::vector<T> deserialize_vector() {
        size_t size = deserialize_u64();
        std::vector<T> value;
        value.reserve(size);
        for (size_t i = 0; i < size; i++)
            value.push_back(deserialize<T>());
        return value;
    }

private:
    const std::vector<uint8_t>& m_Data;
    size_t m_Offset;

public:
    template <typename T>
    T deserialize()
    {
        T value;
        deserialize(value);
        return value;
    }
    template <typename T>
    void deserialize(T& value)
    {
        value = *(T*)&m_Data[m_Offset];
        m_Offset += sizeof(value);
    }
};

template <typename T>
struct deque {
private:
    T* head_{nullptr};
    T* tail_{nullptr};
    size_t size_{0};
public:

    [[nodiscard]] size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] bool is_empty() const noexcept {
        return head_ == nullptr;
    }

    const T* front() const noexcept {
        return head_;
    }

    const T* back() const noexcept {
        return tail_;
    }

    // returns null if failed
    T* remove(T* val) noexcept {
        if (is_empty()) {
            return nullptr;
        }
        if (front() == val) {
            return pop_front();
        } else if (back() == val) {
            return pop_back();
        }

        for (T* n{head_->next}; n; n->next) {
            if (n == val) {
                if (val->next) {
                    val->next->prev = val->prev;
                }
                if (val->prev) {
                    val->prev->next = val->next;
                }

                val->next = val->prev = nullptr;
                return val;
            }
        }
        return nullptr;
    }

    T* erase(size_t index) noexcept {
        return nullptr;
    }

    T* pop_front() noexcept {
        if (is_empty()) {
            return nullptr;
        }
        T* old = head_;
        head_ = head_->next;
        if (head_ == nullptr) {
            tail_ = nullptr;
        } else {
            head_->prev = nullptr;
        }
        return old;
    }

    T* pop_back() noexcept {
        if (is_empty()) {
            return nullptr;
        }
        T* old = tail_;
        tail_ = tail_->prev;
        if (tail_ == nullptr) {
            head_ = nullptr;
        } else {
            tail_->next = nullptr;
        }
        return old;
    }

    void push_back(T* val) {
        val->prev = tail_;
        val->next = nullptr;
        if (is_empty()) {
            head_ = tail_ = val;
        } else {
            tail_->next = val;
            tail_ = val;
        }
    }

    void push_front(T* val) {
        val->prev = nullptr;
        val->next = head_;
        if (is_empty()) {
            head_ = tail_ = val;
        } else {
            head_->next = val;
            head_ = val;
        }
    }
};

template <typename T>
struct free_list_t {
    utl::deque<T> deque;
    T* head{nullptr};
    T* first_free{nullptr};
    size_t min_stack_pos{~static_cast<size_t>(0)};
    size_t count{0};
    size_t free_count{0};

    template<typename allocator_t>
    T* alloc(arena_t* arena) {
        count += 1;
        if (first_free) { 
            T* node {nullptr};  
            node_pop(node, first_free);
            
            new (node) T();          

            deque.push_back(head);
            // node_push(node, head);

            free_count -= 1;
            return node;
        } else {
            // if (allocator.get_marker() < min_stack_pos) {
            //     min_stack_pos = allocator.get_marker();
            // }
            T* node = (T*)arena_alloc(arena, sizeof(T));// allocator.alloc(sizeof(T));
            new (node) T();
            
            // node_push(node, head);
            deque.push_back(node);

            return node;
        }
    }

    void free_index(size_t _i) noexcept {
        T* c = head;
        for (size_t i = 0; i < _i && c; i++, node_next(c)) {
        }
        if (c) {
            free(c);
        }
    }

    // calls T dtor
    void free(T*& o) noexcept {
        if (o == nullptr) { return; }

        T* n = deque.remove(o);
        if (n) {
            node_push(n, first_free);
            o->~T();
            o = nullptr;
            count -= 1;
            free_count += 1;
        }
        return;

        if (head == o) { // if node to free is first
            T* t{nullptr};
            // node_pop(t, head);

            node_push(t, first_free);
            o->~T();
            o = nullptr;
            count -= 1;
            free_count += 1;
            return;
        }
        T* c = head; // find node and free it, while keeping list connected
        while(c && c->next != o) {
            node_next(c);
        }
        if (c && c->next == o) {
            T* t = c->next;
            c->next = c->next->next;
            
            node_push(t, first_free);
            o->~T();
            o = nullptr;
            count -= 1;
            free_count += 1;
        }
    }
};


struct load_results_t {
    gfx::mesh_view_t* meshes{nullptr};
    size_t count{0};
};

inline auto 
make_v3f(const auto& v) -> v3f {
    return {v.x, v.y, v.z};
}

inline load_results_t
pooled_load_mesh(
    arena_t* arena,
    pool_t<gfx::vertex_t>* vertices, 
    pool_t<u32>* indices, 
    std::string_view path
) {
    
}

namespace anim {

using bone_id_t = i32;

struct skeleton_bone_t {
    u64 name_hash{"invalid"_sid};
    bone_id_t parent{-1};
    m44 offset{1.0f};
};

struct skeleton_t {
    static constexpr u64 max_bones_() { return 256; }
    skeleton_bone_t bones[256];
    u64          bone_count{0};

    constexpr skeleton_t& add_bone(skeleton_bone_t&& bone) {
        assert(bone_count < max_bones());
        bones[bone_count++] = std::move(bone);
        return *this;
    }

    constexpr [[nodiscard]] bone_id_t 
    find_bone_id(string_id_t bone_hash) const {
        for (i32 i = 0; i < bone_count; i++) {
            if (bones[i].name_hash == bone_hash) return i;
        }
        return -1;
    }
    
    constexpr [[nodiscard]] const skeleton_bone_t& 
    find_bone(string_id_t bone_name) const {
        return bones[find_bone_id(bone_name)];
    }

    void load(const std::string& path);

    constexpr u64 max_bones() const noexcept { return max_bones_(); }
};

struct skeletal_mesh_asset_t {
    gfx::skinned_vertex_t* vertices{0};
    u64 vertex_count{0};
    gfx::material_info_t* material_info{0};
};

struct skeletal_model_asset_t {
    skeleton_t skeleton{};
    skeletal_mesh_asset_t* meshes{0};
    u64 mesh_count{0};
};

template <typename T>
struct keyframe {
    f32 time{0.0f};
    T value{};
};

template <typename T>
using anim_pool_t = keyframe<T>[512];

struct bone_timeline_t {
    u64 position_count{0};
    u64 rotation_count{0};
    u64 scale_count{0};
    anim_pool_t<v3f>        positions;
    anim_pool_t<glm::quat>  rotations;
    anim_pool_t<v3f>        scales;

    m44 transform{};
    char name[1024]{};
    bone_id_t id{};

    bone_timeline_t(std::string_view pname, bone_id_t pID, const aiNodeAnim* channel);

    size_t get_index(auto& pool, size_t pool_count, f32 time) const {
        for (size_t i = 0; i < pool_count; i++) {
            if (time < pool[i].time) {
                return i;
            }
        }
        assert(0);
        return std::numeric_limits<size_t>::max();
    }
};

struct animation_t {
    struct node_t {
        std::optional<bone_timeline_t> bone;
        bone_id_t parent{-1};
        m44 transform{1.0f};
        m44 offset{1.0f};
    };

    char name[256]{};
    f32 duration{0.0f};
    i32 ticks_per_second{24};

    u64 node_count{0};
    std::array<node_t, skeleton_t::max_bones_()> nodes;
};

struct animator_t {
    pool_t<m44> matrices;

};

template <typename T>
inline void
anim_pool_add_time_value(anim_pool_t<T>* pool, size_t* pool_size, f32 time, T val) {
    auto* time_value = pool[0][(*pool_size)++];
    *time_value = keyframe<T>{
        .time = time,
        .value = val
    };
}

template <typename T>
inline T
anim_pool_get_time_value(anim_pool_t<T>* pool, f32 time) {
    if (pool->count == 0) {
        return T{0.0f};
    }

    if (time <= 0.0f) {
        return (*pool)[0].value;
    } else if (time >= (*pool)[pool->count-1].time) {
        return (*pool)[pool->count-1].value;
    }

    for (size_t i = 0; i < pool->count - 1; i++) {
        if (pool->data()[i].time <= time && time <= pool->data()[i+1].time) {
            const auto d = (pool->data()[i+1].time - pool->data()[i].time);
            const auto p = time - pool->data()[i].time;
            if constexpr (std::is_same_v<T, glm::quat>) {
                return glm::slerp(pool[0][i], pool[0][i+1], p/d);
            } else {
                return glm::mix(
                    pool->data()[i].value,
                    pool->data()[i+1].value,
                    p/d
                );
            }
        }
    }

    return T{};
}


}; // namespace anim


}; // namespace utl

namespace entity {
    inline static constexpr u64 max_entities = 500'000;
    inline static constexpr u64 max_components = 64;

    using entity_t = u64;
    using component_id_t = u64;
    using component_list_t = std::array<u8, max_components>;

    struct component_base_t {
        [[nodiscard]] static u64 type_id() noexcept {
            local_persist u64 id_{0};
            return id_++;
        }
        [[nodiscard]] virtual u64  id() const noexcept = 0;
    };

    template <typename Comp>
    struct component_t : public component_base_t {
        using pool_t = utl::pool_t<Comp>;

        [[nodiscard]] static u64 type_id() noexcept {
            static size_t i{component_base_t::type_id()};
            return i;
        }

        [[nodiscard]] u64 id() const noexcept override final {
            return component_t<Comp>::type_id();
        }
    };

    // [
    //  [base*, ...]
    // ]

    using entity_pool_t = utl::pool_t<component_list_t>;
    using component_pool_t = utl::pool_t<std::array<component_base_t*, max_components>>;

    inline static constexpr entity_t invalid = 0xffffffffffffffff;
    inline static constexpr entity_t dead    = 0xefffffffffffffff;
    
    struct world_t {
        arena_t arena;
        entity_pool_t* entities;
        component_pool_t* components;
    };

    inline world_t*
    world_init(
        arena_t* arena, 
        size_t bytes
    ) noexcept {
        assert(bytes > (max_entities * sizeof(component_list_t)*2));
        world_t* w =  arena_alloc_ctor<world_t>(arena, 1, world_t{});
        w->arena = arena_sub_arena(arena, bytes);
        w->entities =  arena_alloc_ctor<entity_pool_t>(&w->arena, 1, &w->arena, max_entities);
        w->components =  arena_alloc_ctor<component_pool_t>(&w->arena, 1, &w->arena, max_entities);
        
        return w;
    }

    inline entity_t
    world_create_entity(
        world_t* world
    ) noexcept {
        entity_t e = world->entities->count;
        world->entities->create(1);
        world->components->create(1);
        return e;
    }

    template <typename T, typename ... Args>
    inline T*
    world_emplace_component(
        world_t* world,
        entity_t e,
        Args&& ... args
    ) noexcept {
        T* c = arena_alloc_ctor<T>(&world->arena, 1, std::forward<Args>(args)...);
        world_add_component(world, e, c);
        return c;
    }

    inline void
    world_add_component(
        world_t* world, 
        entity_t e,
        component_base_t* component
    ) noexcept {
        world->entities[0][e][component->id()] = 1;
        world->components[0][e][component->id()] = component;
    }

    template <typename T>
    inline T* 
    world_get_component(
        world_t* world, 
        entity_t e
    ) noexcept {
        if (world->entities[0][e][T::type_id()]) {
            return (T*)world->components[0][e][T::type_id()];
        }
        return nullptr;
    }
};
