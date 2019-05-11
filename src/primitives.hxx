#pragma once

#define GLM_FORCE_CXX17
#include <glm/glm.hpp>

namespace primitives {
struct sphere final {
    glm::vec3 center{0};
    float radius{1.f};

    std::size_t material_index;

    template<class T>
    sphere(T &&center, float radius, std::size_t material_index) noexcept
        : center{std::forward<T>(center)}, radius{radius}, material_index{material_index} { }
};

struct ray final {
    glm::vec3 origin;
    glm::vec3 direction;

    ray() = default;

    template<class T1, class T2>
    ray(T1 &&origin, T2 &&direction) noexcept : origin{std::forward<T1>(origin)}, direction{std::forward<T2>(direction)} { }

    glm::vec3 unit_direction() const noexcept { return glm::normalize(direction); }

    glm::vec3 point_at(float t) const noexcept { return origin + direction * t; }
};

struct hit final {
    glm::vec3 position{0};
    glm::vec3 normal{0};

    float time{0.f};

    std::size_t material_index;
};
}
