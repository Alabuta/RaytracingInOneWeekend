#include <iostream>

#include <array>

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
        glm::vec3 A;
        glm::vec3 B;

        template<class T1, class T2>
        ray(T1 &&A, T2 &&B) noexcept : A{std::forward<T1>(A)}, B{std::forward<T2>(B)} { }

        glm::vec3 const &origin() const noexcept { return A; }
        glm::vec3 const &direction() const noexcept { return B; }

        glm::vec3 point_at(float t) const noexcept { return A + B * t; }
    };

    template<class T>
    glm::vec3 background_color(T &&ray)
    {
        auto unit_direction = glm::normalize(ray.direction());

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

    for (auto y = 0; y < app::height; ++y) {
        for (auto x = 0; x < app::width; ++x) {
            auto u = x / static_cast<float>(app::width);
            auto v = y / static_cast<float>(app::height);

            app::ray ray{origin, lower_left_corner + horizontal * u + vertical * (1.f - v)};
            auto color = app::background_color(ray);

            auto r = static_cast<std::uint8_t>(255.f * color.x);
            auto g = static_cast<std::uint8_t>(255.f * color.y);
            auto b = static_cast<std::uint8_t>(255.f * color.z);

            glm::vec<3, std::uint8_t> rgb{r, g, b};

            file.write(reinterpret_cast<char const*>(glm::value_ptr(rgb)), sizeof(rgb));
        }
    }

    std::cout << "Done!\n"; 
}

