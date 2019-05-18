#pragma once

#ifdef CUDA_IMPL
    #include <variant/variant.h>
#else
    #include <variant>
#endif

#include "math.hxx"

namespace material {
struct lambert final {
    math::vec3 albedo{1};

    lambert() = default;

    template<class T>
    CUDA_HOST_DEVICE lambert(T &&albedo) noexcept : albedo{std::forward<T>(albedo)} { }
};

struct metal final {
    math::vec3 albedo{1};
    float roughness{0};

    metal() = default;

    template<class T>
    CUDA_HOST_DEVICE metal(T &&albedo, float roughness) noexcept : albedo{std::forward<T>(albedo)}, roughness{roughness} { }
};

struct dielectric final {
    math::vec3 albedo{1};
    float refraction_index{1};

    dielectric() = default;

    template<class T>
    CUDA_HOST_DEVICE dielectric(T &&albedo, float refraction_index) noexcept : albedo{std::forward<T>(albedo)}, refraction_index{refraction_index} { }
};

#ifdef CUDA_IMPL
    using types = variant::variant<lambert, metal, dielectric>;

    struct surface_response final {
        math::ray ray;
        math::vec3 attenuation;
        bool valid{false};
    };
#else
    using types = std::variant<lambert, metal, dielectric>;
#endif
}
