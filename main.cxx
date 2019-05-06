#include <iostream>
#include <vector>
#include <optional>
#include <random>
#include <execution>
#include <numeric>
#include <chrono>
#include <cmath>
#include <variant>

#include <string>
#include <string_view>
using namespace std::string_literals;
using namespace std::string_view_literals;

#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

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


namespace raytracer {
struct sphere final {
    glm::vec3 center{0};
    float radius{1.f};

    std::size_t material_index;

    template<class T>
    sphere(T &&center, float radius, std::size_t material_index) noexcept
        : center{std::forward<T>(center)}, radius{radius}, material_index{material_index} { }
};

struct lambert final {
    glm::vec3 albedo{1};

    lambert() = default;

    template<class T>
    lambert(T &&albedo) noexcept : albedo{std::forward<T>(albedo)} { }
};

struct metal final {
    glm::vec3 albedo{1};
    float roughness{0};

    metal() = default;

    template<class T>
    metal(T &&albedo, float roughness) noexcept : albedo{std::forward<T>(albedo)}, roughness{roughness} { }
};

struct dielectric final {
    glm::vec3 albedo{1};
    float refraction_index{1};

    dielectric() = default;

    template<class T>
    dielectric(T &&albedo, float refraction_index) noexcept : albedo{std::forward<T>(albedo)}, refraction_index{refraction_index} { }
};

using material = std::variant<lambert, metal, dielectric>;

struct data final {
    static auto constexpr bounces_number = 64u;

    std::random_device random_device;
    std::mt19937 generator;

    std::vector<raytracer::sphere> spheres;

    std::vector<material> materials;

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

        if constexpr (std::is_same_v<type, lambert>) {
            auto random_direction = raytracer::random_in_unit_sphere(generator);
            auto target = position + normal + random_direction;

            auto scattered_ray = raytracer::ray{position, target - position};
            auto attenuation = material.albedo;

            return std::pair{scattered_ray, attenuation};
        }

        else if constexpr (std::is_same_v<type, metal>) {
            auto reflected = glm::reflect(ray.unit_direction(), normal);

            auto random_direction = raytracer::random_in_unit_sphere(generator);

            auto scattered_ray = raytracer::ray{position, reflected + random_direction * material.roughness};
            auto attenuation = material.albedo;

            if (glm::dot(scattered_ray.direction, normal) > 0.f)
                return std::pair{scattered_ray, attenuation};

            return { };
        }

        else if constexpr (std::is_same_v<type, dielectric>) {
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
std::optional<hit> intersect(T1 &&ray, T2 &&sphere)
{
    auto constexpr kMAX = std::numeric_limits<float>::max();
    auto constexpr kMIN = .008f;

    auto oc = ray.origin - sphere.center;

    auto a = glm::dot(ray.direction, ray.direction);
    auto b = glm::dot(oc, ray.direction);
    auto c = glm::dot(oc, oc) - sphere.radius * sphere.radius;

    auto discriminant = b * b - a * c;

    if (discriminant > 0.f) {
        float temp = (-b - std::sqrt(b * b - a * c)) / a;

        if (temp < kMAX && temp > kMIN) {
            auto position = ray.point_at(temp);

            return raytracer::hit{
                position,
                (position - sphere.center) / sphere.radius,
                temp,
                sphere.material_index
            };
        }

        temp = (-b + std::sqrt(b * b - a * c)) / a;

        if (temp < kMAX && temp > kMIN) {
            auto position = ray.point_at(temp);

            return raytracer::hit{
                position,
                (position - sphere.center) / sphere.radius,
                temp,
                sphere.material_index
            };
        }
    }

    /*bool intersected = glm::intersectRaySphere(ray.origin, ray.unit_direction(), sphere.center, sphere.radius, hit.position, hit.normal);

    if (intersected)
        return hit;*/

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

    glm::vec3 origin;
    glm::vec3 lower_left_corner;

    glm::vec3 horizontal;
    glm::vec3 vertical;

    float aspect{1.f};
    float vFOV{glm::radians(72.f)};

    camera(glm::vec3 position, glm::vec3 lookat, glm::vec3 up, float aspect, float vFOV) noexcept : aspect{aspect}, vFOV{vFOV}
    {
        auto theta = glm::radians(vFOV) / 2.f;

        auto height = std::tan(theta);
        auto width = height * aspect;

        origin = position;

        auto w = glm::normalize(position - lookat);
        auto u = glm::normalize(glm::cross(up, w));
        auto v = glm::normalize(glm::cross(w, u));

        //lower_left_corner = glm::vec3{-width, -height, -1.f};
        lower_left_corner = origin - width * u - height * v - w;

        horizontal = 2.f * u * width;
        vertical = 2.f * v * height;
    }

    ray ray(float u, float v) const noexcept
    {
        return {origin, lower_left_corner + horizontal * u + vertical * (1.f - v) - origin};
    }
};
}

namespace app {
struct data final {
    static auto constexpr sampling_number{16u};

    std::uint32_t width{256u};
    std::uint32_t height{128u};

    std::random_device random_device;
    std::mt19937 generator;

    data() : generator{random_device()} { }
};

template<class T>
glm::vec3 constexpr gamma_correction(T &&color)
{
    auto constexpr gamma = glm::vec3{1.f / 2.2f};

    return glm::pow(std::forward<T>(color), gamma);
}

glm::vec3 background_color(float t)
{
    return glm::mix(glm::vec3{1}, glm::vec3{.5, .7, 1}, t);
}

template<class T>
std::optional<raytracer::hit> hit_world(std::vector<raytracer::sphere> const &spheres, T &&ray)
{
    for (auto &&sphere : spheres)
        if (auto hit = intersect(ray, sphere); hit)
            return hit;

    return { };
}

template<class T>
glm::vec3 color(raytracer::data &raytracer_data, T &&ray)
{
    glm::vec3 attenuation{1.f};

    auto scattered_ray = std::forward<T>(ray);
    glm::vec3 energy_absorption{0.f};

    for (auto bounce = 0u; bounce < raytracer_data.bounces_number; ++bounce) {
        if (auto hit = hit_world(raytracer_data.spheres, scattered_ray); hit) {
            if (auto scattered = raytracer::apply_material(raytracer_data, scattered_ray, hit.value()); scattered) {
                std::tie(scattered_ray, energy_absorption) = *scattered;

                attenuation *= energy_absorption;
            }

            else return glm::vec3{0};
        }

        else return app::background_color(.5f * scattered_ray.unit_direction().y + 1.f) * attenuation;
    }

    return glm::vec3{0};
}
}

int main()
{
    raytracer::data raytracer_data;

    raytracer_data.materials.emplace_back(raytracer::lambert{glm::vec3{.1, .2, .5}});
    raytracer_data.materials.emplace_back(raytracer::metal{glm::vec3{.8, .6, .2}, 0});
    raytracer_data.materials.emplace_back(raytracer::dielectric{glm::vec3{1}, 1.5f});
    raytracer_data.materials.emplace_back(raytracer::lambert{glm::vec3{.8, .8, 0}});

    raytracer_data.spheres.emplace_back(glm::vec3{0, 0, -1}, .5f, 0);
    raytracer_data.spheres.emplace_back(glm::vec3{+1, 0, -1}, .5f, 1);
    raytracer_data.spheres.emplace_back(glm::vec3{-1, 0, -1}, .5f, 2);
    raytracer_data.spheres.emplace_back(glm::vec3{0, -100.5, -1}, 100.f, 3);

    app::data app_data;

    auto random_distribution = std::uniform_real_distribution{0.f, 1.f};

    raytracer::camera camera{
        glm::vec3{-2, 2, 1}, glm::vec3{0, 0, -1}, glm::vec3{0, 1, 0},
        static_cast<float>(app_data.width) / static_cast<float>(app_data.height), 30.f
    };

    std::vector<glm::vec3> multisampling_texels(app_data.sampling_number, glm::vec3{0});

    std::vector<glm::vec<3, std::uint8_t>> data(static_cast<std::size_t>(app_data.width) * app_data.height);

    for (auto y = 0u; y < app_data.height; ++y) {
        auto v = static_cast<float>(y) / static_cast<float>(app_data.height);

        for (auto x = 0u; x < app_data.width; ++x) {
            auto u = static_cast<float>(x) / static_cast<float>(app_data.width);

            std::generate(std::execution::par, std::begin(multisampling_texels), std::end(multisampling_texels), [&] ()
            {
                auto _u = u + random_distribution(raytracer_data.generator) / static_cast<float>(app_data.width);
                auto _v = v + random_distribution(raytracer_data.generator) / static_cast<float>(app_data.height);

                return app::color(raytracer_data, camera.ray(_u, _v));
            });

            auto color = std::reduce(std::execution::par, std::begin(multisampling_texels), std::end(multisampling_texels), glm::vec3{0});

            color /= static_cast<float>(app_data.sampling_number);

            color = app::gamma_correction(color);

            auto &&rgb = data[x + static_cast<std::size_t>(app_data.width) * y];

            rgb.r = static_cast<std::uint8_t>(255.f * color.x);
            rgb.g = static_cast<std::uint8_t>(255.f * color.y);
            rgb.b = static_cast<std::uint8_t>(255.f * color.z);
        }
    }

    fs::path path{"image.ppm"sv};

    std::ofstream file{path, std::ios::out | std::ios::trunc | std::ios::binary};

    if (file.bad() || file.fail())
        return 1;

    file << "P6\n"s << app_data.width << " "s << app_data.height << "\n255\n"s;

    file.write(reinterpret_cast<char const *>(std::data(data)), std::size(data) * sizeof(decltype(data)::value_type));
}

