#pragma once

#ifndef CUDA_IMPL
    #include <variant>
#endif

#include "math.hxx"

namespace material {
struct abstract {
    virtual ~abstract() = default;
};

struct lambert final : public material::abstract {
    math::vec3 albedo{1};

    lambert() = default;

    template<class T>
    lambert(T &&albedo) noexcept : albedo{std::forward<T>(albedo)} { }
};

struct metal final : public material::abstract {
    math::vec3 albedo{1};
    float roughness{0};

    metal() = default;

    template<class T>
    metal(T &&albedo, float roughness) noexcept : albedo{std::forward<T>(albedo)}, roughness{roughness} { }
};

struct dielectric final : public material::abstract {
    math::vec3 albedo{1};
    float refraction_index{1};

    dielectric() = default;

    template<class T>
    dielectric(T &&albedo, float refraction_index) noexcept : albedo{std::forward<T>(albedo)}, refraction_index{refraction_index} { }
};

#ifdef CUDA_IMPL
    using types = abstract;
#else
    using types = std::variant<lambert, metal, dielectric>;
#endif
}
