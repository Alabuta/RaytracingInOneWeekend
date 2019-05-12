#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <cmath>

#include "math.hxx"
#include "raytracer.hxx"


namespace raytracer {
class camera final {
public:

    float aspect{1.f};
    float vFOV{math::radians(72.f)};

    float lens_radius{1.f};

    camera(math::vec3 position, math::vec3 lookat, math::vec3 up, float aspect, float vFOV, float aperture, float focus_distance) noexcept
        : aspect{aspect}, vFOV{vFOV}, lens_radius{aperture / 2.f}, generator{random_device()}, origin{position}
    {
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

    math::ray ray(float u, float v) noexcept
    {
        auto random_direction = raytracer::random_in_unit_sphere(generator) * lens_radius;

        math::vec3 offset{u * random_direction.x , v * random_direction.y, 0};

        return {origin + offset, lower_left_corner + horizontal * u + vertical * (1.f - v) - origin - offset};
    }

private:

    std::random_device random_device;
    std::mt19937 generator;

    math::vec3 origin;

    math::vec3 lower_left_corner;

    math::vec3 horizontal;
    math::vec3 vertical;
};
}