#include <iostream>

#include <array>

#include <string>
#include <string_view>

using namespace std::string_literals;
using namespace std::string_view_literals;

#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

auto const magic_number{"P6"s};

auto constexpr width{3u};
auto constexpr height{2u};

using rgb_t = std::array<std::uint8_t, 3>;

int main()
{
    fs::path path{"image.ppm"sv};

    std::ofstream file{path, std::ios::out | std::ios::trunc | std::ios::binary};

    if (file.bad() || file.fail())
        return 1;

    file << magic_number << '\n' << width << " "s << height << "\n255\n"s;

    std::array<std::array<rgb_t, width>, height> data{{
        {
            rgb_t{255, 0, 0}, rgb_t{0, 255, 0}, rgb_t{0, 0, 255}
        },
        {
            rgb_t{255, 255, 0}, rgb_t{255, 255, 255}, rgb_t{0, 0, 0}
        }
    }};

    for (auto &&row : data) {
        for (auto &&column : row) {
            file.write(reinterpret_cast<char const*>(std::data(column)), std::size(column) * sizeof(rgb_t));

            file.put(' ');
        }

        file.put('\n');
    }

    /*for (auto x = 0; x < width; ++x) {
        for (auto y = 0; y < height; ++y) {
            ;
        }
    }*/

    std::cout << "Done!\n"; 
}

