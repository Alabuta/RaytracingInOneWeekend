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

#include "primitives.hxx"
#include "material.hxx"

namespace raytracer {
struct data final {
    static auto constexpr bounces_number = 64u;

    std::random_device random_device;
    std::mt19937 generator;

    std::vector<primitives::sphere> spheres;

    std::vector<material::types> materials;

    data() : generator{random_device()} { }
};

glm::vec3 random_in_unit_sphere(std::mt19937 &generator)
{
    static auto random_distribution = std::uniform_real_distribution{-1.f, +1.f};

    glm::vec3 vector;

    do {
        vector = glm::vec3{random_distribution(generator), random_distribution(generator), random_distribution(generator)};
    } while (glm::length(vector) > 1.f);

    return vector;
}

float schlick_reflection_probability(float refraction_index, float cosine_theta)
{
    auto reflection_coefficient = std::pow((1.f - refraction_index) / (1.f + refraction_index), 2);

    return reflection_coefficient + (1.f - reflection_coefficient) * std::pow(1.f - cosine_theta, 5);
}

template<class T1, class T2, typename std::enable_if_t<std::is_same_v<primitives::sphere, std::decay_t<T2>>>...>
std::optional<primitives::hit> intersect(T1 &&ray, T2 &&sphere, float time_min, float time_max)
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

            return primitives::hit{
                position,
                (position - sphere.center) / sphere.radius,
                temp,
                sphere.material_index
            };
        }

        temp = (-b + std::sqrt(b * b - a * c)) / a;

        if (temp < time_max && temp > time_min) {
            auto position = ray.point_at(temp);

            return primitives::hit{
                position,
                (position - sphere.center) / sphere.radius,
                temp,
                sphere.material_index
            };
        }
    }

    return { };
}

template<class T>
std::optional<primitives::hit> hit_world(std::vector<primitives::sphere> const &spheres, T &&ray)
{
    auto constexpr kMAX = std::numeric_limits<float>::max();
    auto constexpr kMIN = .008f;

    std::vector<std::optional<primitives::hit>> hits(std::size(spheres));

    std::transform(std::execution::seq, std::cbegin(spheres), std::cend(spheres), std::begin(hits), [&ray, kMIN, kMAX] (auto &&sphere)
    {
        return raytracer::intersect(ray, sphere, kMIN, kMAX);
    });

    auto it_end = std::stable_partition(std::execution::par, std::begin(hits), std::end(hits), [] (auto &&hit)
    {
        return hit.has_value();
    });

    auto it_hit = std::min_element(std::execution::par_unseq, std::begin(hits), it_end, [] (auto &&lhs, auto &&rhs)
    {
        return lhs->time < rhs->time;
    });

    return it_hit != it_end ? *it_hit : std::optional<primitives::hit>{ };
}

template<class T1, class T2>
std::optional<std::pair<primitives::ray, glm::vec3>> apply_material(raytracer::data &raytracer_data, T1 &&ray, T2 &&hit)
{
    auto &&[position, normal, time, material_index] = hit;

    auto &&generator = raytracer_data.generator;
    auto const &materials = raytracer_data.materials;

    return std::visit([&] (auto &&material) -> std::optional<std::pair<primitives::ray, glm::vec3>>
    {
        using type = std::decay_t<decltype(material)>;

        if constexpr (std::is_same_v<type, material::lambert>)
        {
            auto random_direction = raytracer::random_in_unit_sphere(generator);
            auto target = position + normal + random_direction;

            auto scattered_ray = primitives::ray{position, target - position};
            auto attenuation = material.albedo;

            return std::pair{scattered_ray, attenuation};
        }

        else if constexpr (std::is_same_v<type, material::metal>)
        {
            auto reflected = glm::reflect(ray.unit_direction(), normal);

            auto random_direction = raytracer::random_in_unit_sphere(generator);

            auto scattered_ray = primitives::ray{position, reflected + random_direction * material.roughness};
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

            primitives::ray scattered_ray;

            if (random_distribution(generator) < reflection_probability) {
                auto reflected = glm::reflect(ray.unit_direction(), normal);
                scattered_ray = primitives::ray{position, reflected};
            }

            else scattered_ray = primitives::ray{position, refracted};

            return std::pair{scattered_ray, attenuation};
        }

        else static_assert(std::false_type{}, "unsupported material type");

    }, materials[material_index]);
}
}