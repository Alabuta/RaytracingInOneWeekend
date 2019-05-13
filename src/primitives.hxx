#pragma once

#include "math.hxx"

namespace primitives {
struct sphere final {
    math::vec3 center{0};
    float radius{1.f};

    std::size_t material_index;

    sphere() = default;

    template<class T>
    CUDA_HOST_DEVICE sphere(T &&center, float radius, std::size_t material_index) noexcept
        : center{std::forward<T>(center)}, radius{radius}, material_index{material_index} { }
};

struct hit final {
    math::vec3 position{0};
    math::vec3 normal{0};

    float time{0.f};

    std::size_t material_index;
};
}
