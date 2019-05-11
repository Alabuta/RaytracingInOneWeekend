#pragma once

//#define GLM_FORCE_CUDA
//#define GLM_FORCE_CXX17
//#include <glm/glm.hpp>


template<class T>
struct RGB final {
    T r, g, b;
};


namespace math {
template<class T>
struct _vec3 final {
    using type = T;

    T x, y, z;

    _vec3() = default;

    constexpr _vec3(T x, T y, T z) noexcept : x{x}, y{y}, z{z} { }
    constexpr _vec3(T value) noexcept : x{value}, y{value}, z{value} { }
};

using vec3 = _vec3<float>;
using u8vec3 = _vec3<std::uint8_t>;
}
