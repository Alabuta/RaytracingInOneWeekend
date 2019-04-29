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

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

auto const magic_number{"P6"s};

auto constexpr width{256u};
auto constexpr height{256u};


int main()
{
    fs::path path{"image.ppm"sv};

    std::ofstream file{path, std::ios::out | std::ios::trunc | std::ios::binary};

    if (file.bad() || file.fail())
        return 1;

    file << magic_number << '\n' << width << " "s << height << "\n255\n"s;

    for (auto x = 0; x < width; ++x) {
        for (auto y = 0; y < height; ++y) {
            auto r = static_cast<std::uint8_t>(255.f * x / static_cast<float>(width));
            auto g = static_cast<std::uint8_t>(255.f * y / static_cast<float>(height));
            auto b = static_cast<std::uint8_t>(255.f * .2f);

            glm::vec<3, std::uint8_t> rgb{r, g, b};

            file.write(reinterpret_cast<char const*>(glm::value_ptr(rgb)), sizeof(rgb));
        }
    }

    std::cout << "Done!\n"; 
}

