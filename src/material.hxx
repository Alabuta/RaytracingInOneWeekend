#pragma once

#include <variant>

#include "math.hxx"

namespace material {
struct lambert final {
    math::vec3 albedo{1};

    lambert() = default;

    template<class T>
    lambert(T &&albedo) noexcept : albedo{std::forward<T>(albedo)} { }
};

struct metal final {
    math::vec3 albedo{1};
    float roughness{0};

    metal() = default;

    template<class T>
    metal(T &&albedo, float roughness) noexcept : albedo{std::forward<T>(albedo)}, roughness{roughness} { }
};

struct dielectric final {
    math::vec3 albedo{1};
    float refraction_index{1};

    dielectric() = default;

    template<class T>
    dielectric(T &&albedo, float refraction_index) noexcept : albedo{std::forward<T>(albedo)}, refraction_index{refraction_index} { }
};

using types = std::variant<lambert, metal, dielectric>;
}
