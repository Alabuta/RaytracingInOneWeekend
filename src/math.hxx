#pragma once

//#define GLM_FORCE_CUDA
//#define GLM_FORCE_CXX17
//#include <glm/glm.hpp>


#ifdef CUDA_VERSION
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__
#else
    #define CUDA_HOST
    #define CUDA_DEVICE
#endif


namespace math {
    template<std::size_t N, class T>
    struct vec final {
        using value_type = T;
        static auto constexpr size{N};
    };

    template<class T>
    struct vec<3, T> final {
        T x, y, z;

        vec() = default;

        CUDA_HOST CUDA_DEVICE constexpr vec(T x, T y, T z) noexcept : x{x}, y{y}, z{z} { }
        CUDA_HOST CUDA_DEVICE constexpr vec(T value) noexcept : x{value}, y{value}, z{value} { }
    };

    using vec3 = vec<3, float>;
    using u8vec3 = vec<3, std::uint8_t>;
}
