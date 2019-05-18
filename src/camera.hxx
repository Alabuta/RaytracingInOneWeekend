#pragma once

#include <vector>
#ifndef CUDA_IMPL
    #include <random>
#endif
#include <numeric>
#include <cmath>

#include "main.hxx"
#include "math.hxx"

#ifndef CUDA_IMPL
    #include "raytracer.hxx"
#endif


namespace raytracer {
class camera final {
public:

    camera() = default;

    CUDA_HOST_DEVICE camera(math::vec3 position, math::vec3 lookat, math::vec3 up, float aspect, float vFOV, float aperture, float focus_distance) noexcept
        : aspect{aspect}, vFOV{vFOV}, lens_radius{aperture / 2.f}, origin{position}
    {
    #ifndef CUDA_IMPL
        generator = std::mt19937{random_device()};
    #endif

        auto theta = math::radians(vFOV) / 2.f;

        auto height = std::tan(theta);
        auto width = height * aspect;

        auto w = math::normalize(position - lookat);
        auto u = math::normalize(math::cross(up, w));
        auto v = math::normalize(math::cross(w, u));

        lower_left_corner = origin - (u * width + v * height + w) * focus_distance;

        horizontal = u * width * focus_distance * 2.f;
        vertical = v * height * focus_distance * 2.f;
    }

    CUDA_DEVICE math::ray ray(float u, float v) noexcept
    {
    #ifdef CUDA_IMPL
        math::vec3 offset{0};
    #else
        auto random_direction = raytracer::random_in_unit_sphere(generator) * lens_radius;

        math::vec3 offset{u * random_direction.x , v * random_direction.y, 0};
    #endif

        return {origin + offset, lower_left_corner + horizontal * u + vertical * (1.f - v) - offset};
    }

private:

#ifndef CUDA_IMPL
    std::random_device random_device;
    std::mt19937 generator;
#endif

    float aspect{1.f};
    float vFOV{72.f};

    float lens_radius{1.f};

    math::vec3 origin;

    math::vec3 lower_left_corner;

    math::vec3 horizontal;
    math::vec3 vertical;
};
}