#include <iostream>

#include <optional>

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

namespace app {
    auto const magic_number{"P6"s};

    auto constexpr width{1024u};
    auto constexpr height{512u};

    struct ray final {
        glm::vec3 origin;
        glm::vec3 direction;

        template<class T1, class T2>
        ray(T1 &&origin, T2 &&direction) noexcept : origin{std::forward<T1>(origin)}, direction{std::forward<T2>(direction)} { }

        glm::vec3 unit_direction() const noexcept { return glm::normalize(direction); }

        glm::vec3 point_at(float t) const noexcept { return origin + direction * t; }
    };

    struct sphere final {
        glm::vec3 center{0};
        float radius{1.f};

        template<class T>
        sphere(T &&center, float radius) noexcept : center{std::forward<T>(center)}, radius{radius} { }
    };

    template<class T1, class T2, typename std::enable_if_t<std::is_same_v<sphere, std::decay_t<T2>>>...>
    std::optional<glm::vec3> intersect(T1 &&ray, T2 &&sphere)
    {
        glm::vec3 position, normal;

        bool intersected = glm::intersectRaySphere(ray.origin, ray.unit_direction(), sphere.center, sphere.radius, position, normal);

        if (intersected)
            return position;

        return { };
    }

    template<class T1, class T2>
    glm::vec3 color(T1 &&ray, T2 &&sphere)
    {
        if (auto position = intersect(ray, sphere); position) {
            return glm::vec3{1, 0, 0};
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

    glm::vec3 origin{0};
    glm::vec3 lower_left_corner{-2, -1, -1};

    glm::vec3 horizontal{4, 0, 0};
    glm::vec3 vertical{0, 2, 0};

    app::sphere sphere{glm::vec3{0, 0, -1}, .5f};

    for (auto y = 0; y < app::height; ++y) {
        for (auto x = 0; x < app::width; ++x) {
            auto u = x / static_cast<float>(app::width);
            auto v = y / static_cast<float>(app::height);

            app::ray ray{origin, lower_left_corner + horizontal * u + vertical * (1.f - v)};
            auto color = app::color(ray, sphere);

            auto r = static_cast<std::uint8_t>(255.f * color.x);
            auto g = static_cast<std::uint8_t>(255.f * color.y);
            auto b = static_cast<std::uint8_t>(255.f * color.z);

            glm::vec<3, std::uint8_t> rgb{r, g, b};

            file.write(reinterpret_cast<char const*>(glm::value_ptr(rgb)), sizeof(rgb));
        }
    }

    std::cout << "Done!\n"; 
}

