#pragma once

#include <iostream>
#include <vector>
#include <optional>
#include <random>
#include <execution>
#include <numeric>
#include <chrono>
#include <cmath>
#include <variant>

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

#include "material.hxx"

namespace raytracer {
struct sphere final {
    glm::vec3 center{0};
    float radius{1.f};

    std::size_t material_index;

    template<class T>
    sphere(T &&center, float radius, std::size_t material_index) noexcept
        : center{std::forward<T>(center)}, radius{radius}, material_index{material_index} { }
};

struct data final {
    static auto constexpr bounces_number = 64u;

    std::random_device random_device;
    std::mt19937 generator;

    std::vector<raytracer::sphere> spheres;

    std::vector<material::types> materials;

    data() : generator{random_device()} { }
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

float schlick_reflection_probability(float refraction_index, float cosine_theta)
{
    auto reflection_coefficient = std::pow((1.f - refraction_index) / (1.f + refraction_index), 2);

    return reflection_coefficient + (1.f - reflection_coefficient) * std::pow(1.f - cosine_theta, 5);
}

template<class T1, class T2>
std::optional<std::pair<raytracer::ray, glm::vec3>> apply_material(raytracer::data &raytracer_data, T1 &&ray, T2 &&hit)
{
    auto &&[position, normal, time, material_index] = hit;

    auto &&generator = raytracer_data.generator;
    auto const &materials = raytracer_data.materials;

    return std::visit([&] (auto &&material) -> std::optional<std::pair<raytracer::ray, glm::vec3>>
    {
        using type = std::decay_t<decltype(material)>;

        if constexpr (std::is_same_v<type, material::lambert>)
        {
            auto random_direction = raytracer::random_in_unit_sphere(generator);
            auto target = position + normal + random_direction;

            auto scattered_ray = raytracer::ray{position, target - position};
            auto attenuation = material.albedo;

            return std::pair{scattered_ray, attenuation};
        }

        else if constexpr (std::is_same_v<type, material::metal>)
        {
            auto reflected = glm::reflect(ray.unit_direction(), normal);

            auto random_direction = raytracer::random_in_unit_sphere(generator);

            auto scattered_ray = raytracer::ray{position, reflected + random_direction * material.roughness};
            auto attenuation = material.albedo;

            if (glm::dot(scattered_ray.direction, normal) > 0.f)
                return std::pair{scattered_ray, attenuation};

            return { };
        }

        else if constexpr (std::is_same_v<type, material::dielectric>)
        {
            auto outward_normal = -normal;
            auto refraction_index = material.refraction_index;;
            auto cosine_theta = glm::dot(ray.unit_direction(), normal);

            if (cosine_theta <= 0.f) {
                outward_normal *= -1.f;
                refraction_index = 1.f / refraction_index;
                cosine_theta *= -1.f;
            }

            //else cosine_theta *= refraction_index;

            //cosine_theta /= glm::length(ray.direction);

            auto attenuation = material.albedo;
            auto refracted = glm::refract(ray.unit_direction(), outward_normal, refraction_index);

            auto reflection_probability = 1.f;

            if (glm::length(refracted) > 0.f)
                reflection_probability = raytracer::schlick_reflection_probability(refraction_index, cosine_theta);

            static auto random_distribution = std::uniform_real_distribution{0.f, 1.f};

            raytracer::ray scattered_ray;

            if (random_distribution(generator) < reflection_probability) {
                auto reflected = glm::reflect(ray.unit_direction(), normal);
                scattered_ray = raytracer::ray{position, reflected};
            }

            else scattered_ray = raytracer::ray{position, refracted};

            return std::pair{scattered_ray, attenuation};
        }

        else static_assert(std::false_type{}, "unsupported material type");

    }, materials[material_index]);
}

template<class T1, class T2, typename std::enable_if_t<std::is_same_v<raytracer::sphere, std::decay_t<T2>>>...>
std::optional<hit> intersect(T1 &&ray, T2 &&sphere, float time_min, float time_max)
{
    auto oc = ray.origin - sphere.center;

    auto a = glm::dot(ray.direction, ray.direction);
    auto b = glm::dot(oc, ray.direction);
    auto c = glm::dot(oc, oc) - sphere.radius * sphere.radius;

    auto discriminant = b * b - a * c;

    if (discriminant > 0.f) {
        float temp = (-b - std::sqrt(b * b - a * c)) / a;

        if (temp < time_max && temp > time_min) {
            auto position = ray.point_at(temp);

            return raytracer::hit{
                position,
                (position - sphere.center) / sphere.radius,
                temp,
                sphere.material_index
            };
        }

        temp = (-b + std::sqrt(b * b - a * c)) / a;

        if (temp < time_max && temp > time_min) {
            auto position = ray.point_at(temp);

            return raytracer::hit{
                position,
                (position - sphere.center) / sphere.radius,
                temp,
                sphere.material_index
            };
        }
    }

    return { };
}

glm::vec3 random_in_unit_sphere(std::mt19937 &generator)
{
    static auto random_distribution = std::uniform_real_distribution{-1.f, +1.f};

    glm::vec3 vector;

    do {
        vector = glm::vec3{random_distribution(generator), random_distribution(generator), random_distribution(generator)};
    } while (glm::length(vector) > 1.f);

    return vector;
}

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

    raytracer::ray ray(float u, float v) noexcept
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