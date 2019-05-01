#include <iostream>
#include <vector>
#include <optional>
#include <random>
#include <execution>
#include <numeric>
#include <chrono>

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


namespace math
{
    struct ray final {
        glm::vec3 origin;
        glm::vec3 direction;

        template<class T1, class T2>
        ray(T1 &&origin, T2 &&direction) noexcept : origin{std::forward<T1>(origin)}, direction{std::forward<T2>(direction)} { }

        glm::vec3 unit_direction() const noexcept { return glm::normalize(direction); }

        glm::vec3 point_at(float t) const noexcept { return origin + direction * t; }
    };

    struct hit final {
        glm::vec3 position{0};
        glm::vec3 normal{0};
        float time{0.f};
    };

    struct sphere final {
        glm::vec3 center{0};
        float radius{1.f};

        template<class T>
        sphere(T &&center, float radius) noexcept : center{std::forward<T>(center)}, radius{radius} { }
    };

    template<class T1, class T2, typename std::enable_if_t<std::is_same_v<sphere, std::decay_t<T2>>>...>
    std::optional<hit> intersect(T1 &&ray, T2 &&sphere)
    {
        math::hit hit;

        bool intersected = glm::intersectRaySphere(ray.origin, ray.unit_direction(), sphere.center, sphere.radius, hit.position, hit.normal);

        if (intersected)
            return hit;

        return { };
    }

    class camera final {
    public:

        glm::vec3 origin;
        glm::vec3 lower_left_corner;

        glm::vec3 horizontal;
        glm::vec3 vertical;
        
        template<class T1, class T2, class T3, class T4>
        camera(T1 &&origin, T2 &&lower_left_corner, T3 &&horizontal, T4 &&vertical) noexcept
            : origin{origin}, lower_left_corner{lower_left_corner}, horizontal{horizontal}, vertical{vertical} { };

        ray ray(float u, float v) const noexcept
        {
            return {origin, lower_left_corner + horizontal * u + vertical * (1.f - v)};
        }
    };
}

namespace app {
    auto const magic_number{"P6"s};

    auto constexpr width{256u};
    auto constexpr height{128u};

    template<class T>
    glm::vec3 color(std::vector<math::sphere> const &spheres, T &&ray)
    {
        for (auto&& sphere : spheres) {
            if (auto hit = intersect(ray, sphere); hit) {
                auto&& [position, normal, time] = *hit;

                return normal * .5f + .5f;
            }
        }

        auto unit_direction = ray.unit_direction();

        auto t = .5f * unit_direction.y + 1.f;

        return glm::mix(glm::vec3{1}, glm::vec3{.5, .7, 1}, t);
    }
}

int main()
{
    fs::path path{"image.ppm"sv};

    std::ofstream file{path, std::ios::out | std::ios::trunc | std::ios::binary};

    if (file.bad() || file.fail())
        return 1;

    file << app::magic_number << '\n' << app::width << " "s << app::height << "\n255\n"s;

    std::random_device random_device;
    std::mt19937 generator{random_device()};

    auto random_distribution = std::uniform_real_distribution{0.f, 1.f};

    math::camera camera{glm::vec3{0}, glm::vec3{-2, -1, -1}, glm::vec3{4, 0, 0}, glm::vec3{0, 2, 0}};

    std::vector<math::sphere> spheres;

    spheres.emplace_back(glm::vec3{0, 0, -1}, .5f);
    spheres.emplace_back(glm::vec3{0, -100.5, -1}, 100.f);

    auto constexpr sampling_number = 32u;

    std::vector<glm::vec3> colors(sampling_number, glm::vec3{0});

    //auto const start = std::chrono::high_resolution_clock::now();

    std::vector<glm::vec<3, std::uint8_t>> data(app::width * app::height);

    for (auto y = 0u; y < app::height; ++y) {
        auto v = static_cast<float>(y) / static_cast<float>(app::height);

        for (auto x = 0u; x < app::width; ++x) {
            auto u = static_cast<float>(x) / static_cast<float>(app::width);

            std::generate(std::execution::par, std::begin(colors), std::end(colors), [&] ()
            {
                auto _u = u + random_distribution(generator) / static_cast<float>(app::width);
                auto _v = v + random_distribution(generator) / static_cast<float>(app::height);

                return app::color(spheres, camera.ray(_u, _v));
            });

            auto color = std::reduce(std::execution::par, std::begin(colors), std::end(colors), glm::vec3{0});

            color /= static_cast<float>(sampling_number);

            auto &&rgb = data[x + static_cast<std::size_t>(app::width) * y];

            rgb.r = static_cast<std::uint8_t>(255.f * color.x);
            rgb.g = static_cast<std::uint8_t>(255.f * color.y);
            rgb.b = static_cast<std::uint8_t>(255.f * color.z);
        }
    }

    file.write(reinterpret_cast<char const*>(std::data(data)), std::size(data) * sizeof(decltype(data)::value_type));

    /*auto const end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "reduce aproach took "s << ms.count() << " ms\n"s;*/
}

