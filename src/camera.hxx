#pragma once

#include <vector>
#include <random>
#include <numeric>
#include <cmath>

#define GLM_FORCE_CXX17
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_GTX_intersect

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "raytracer.hxx"


namespace raytracer {
class camera final {
public:

    float aspect{1.f};
    float vFOV{glm::radians(72.f)};

    float lens_radius{1.f};

    camera(glm::vec3 position, glm::vec3 lookat, glm::vec3 up, float aspect, float vFOV, float aperture, float focus_distance) noexcept
        : aspect{aspect}, vFOV{vFOV}, lens_radius{aperture / 2.f}, generator{random_device()}, origin{position}
    {
        auto theta = glm::radians(vFOV) / 2.f;

        auto height = std::tan(theta);
        auto width = height * aspect;

        auto w = glm::normalize(position - lookat);
        auto u = glm::normalize(glm::cross(up, w));
        auto v = glm::normalize(glm::cross(w, u));

        lower_left_corner = origin - (width * u + height * v + w) * focus_distance;

        horizontal = 2.f * u * width * focus_distance;
        vertical = 2.f * v * height * focus_distance;
    }

    primitives::ray ray(float u, float v) noexcept
    {
        auto random_direction = raytracer::random_in_unit_sphere(generator) * lens_radius;

        glm::vec3 offset{u * random_direction.x , v * random_direction.y, 0};

        return {origin + offset, lower_left_corner + horizontal * u + vertical * (1.f - v) - origin - offset};
    }

private:

    std::random_device random_device;
    std::mt19937 generator;

    glm::vec3 origin;

    glm::vec3 lower_left_corner;

    glm::vec3 horizontal;
    glm::vec3 vertical;
};
}