#include <optional>
#include <execution>
#include <variant>

#include <string_view>
using namespace std::string_view_literals;

#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include "main.hxx"
#include "math.hxx"
#include "raytracer.hxx"
#include "camera.hxx"


extern void cuda_impl(std::uint32_t width, std::uint32_t height, std::vector<math::u8vec3> &image_texels);


namespace app {
struct data final {
    static auto constexpr sampling_number{16u};

#ifdef _DEBUG
    std::uint32_t width{512u};
    std::uint32_t height{256u};
#else
    std::uint32_t width{1920u};
    std::uint32_t height{1080u};
#endif

    std::random_device random_device;
    std::mt19937 generator;

    data() : generator{random_device()} { }
};

template<class T>
math::vec3 gamma_correction(T &&color)
{
    auto constexpr gamma = math::vec3{1.f / 2.2f};

    return math::pow(std::forward<T>(color), gamma);
}

math::vec3 background_color(float t)
{
    return math::mix(math::vec3{1}, math::vec3{.5, .7, 1}, t);
}

template<class T>
math::vec3 color(raytracer::data &raytracer_data, T &&ray)
{
    math::vec3 attenuation{1};

    auto scattered_ray = std::forward<T>(ray);
    math::vec3 energy_absorption{0};

    for (auto bounce = 0u; bounce < raytracer_data.bounces_number; ++bounce) {
        if (auto hit = raytracer::hit_world(raytracer_data.spheres, scattered_ray); hit) {
            if (auto scattered = raytracer::apply_material(raytracer_data, scattered_ray, hit.value()); scattered) {
                std::tie(scattered_ray, energy_absorption) = *scattered;

                attenuation *= energy_absorption;
            }

            else return math::vec3{0};
        }

        else return app::background_color(.5f * scattered_ray.unit_direction().y + 1.f) * attenuation;
    }

    return math::vec3{0};
}

template<class T>
math::u8vec3 normalize_rgb_to_8bit(T &&color)
{
    return {
        static_cast<std::uint8_t>(255.f * color.x),
        static_cast<std::uint8_t>(255.f * color.y),
        static_cast<std::uint8_t>(255.f * color.z)
    };
}

void save_to_file(std::string_view name, app::data const &app_data, std::vector<math::u8vec3> const &texels_data)
{
    fs::path path{name};

    std::ofstream file{path, std::ios::out | std::ios::trunc | std::ios::binary};

    if (file.bad() || file.fail())
        throw std::runtime_error("bad file"s);

    file << "P6\n"s << app_data.width << " "s << app_data.height << "\n255\n"s;

    using texel_type = std::decay_t<decltype(texels_data)>::value_type;

    file.write(reinterpret_cast<char const *>(std::data(texels_data)), std::size(texels_data) * sizeof(texel_type));
}
}

int main()
{
    std::cout << "started... \n"s;

    app::data app_data;

    raytracer::data raytracer_data;

    std::vector<math::u8vec3> output(static_cast<std::size_t>(app_data.width) * app_data.height);

    cuda_impl(app_data.width, app_data.height, output);

    app::save_to_file("image_cuda.ppm"sv, app_data, output);

    return 0;

    raytracer_data.materials.emplace_back(material::lambert{math::vec3{.1, .2, .5}});
    raytracer_data.materials.emplace_back(material::metal{math::vec3{.8, .6, .2}, 0});
    raytracer_data.materials.emplace_back(material::dielectric{math::vec3{1}, 1.5f});
    raytracer_data.materials.emplace_back(material::lambert{math::vec3{.64, .8, .0}});

    raytracer_data.spheres.emplace_back(math::vec3{0, 1, 0}, 1.f, 0);
    raytracer_data.spheres.emplace_back(math::vec3{0, -1000.125f, 0}, 1000.f, 3);
    raytracer_data.spheres.emplace_back(math::vec3{+2, 1, 0}, 1.f, 1);
    raytracer_data.spheres.emplace_back(math::vec3{-2, 1, 0}, 1.f, 2);
    raytracer_data.spheres.emplace_back(math::vec3{-2, 1, 0}, -.99f, 2);

#if 0
    {
        std::random_device random_device;
        std::mt19937 generator{random_device()};

        auto rd_int = std::uniform_int_distribution{0, 3};
        auto rd_real = std::uniform_real_distribution{0.f, 1.f};

        for (auto a = -11; a < 11; ++a) {
            for (auto b = -11; b < 11; ++b) {
                auto material_type_index = rd_int(generator);

                math::vec3 center{.9f * rd_real(generator) + a, .2f, .9f * rd_real(generator) + b};

                if (math::distance(center, math::vec3{0, 1, 0}) < 1.f)
                    continue;

                raytracer_data.spheres.emplace_back(
                    center, .2f, std::size(raytracer_data.materials)
                );

                switch (material_type_index) {
                    case 0:
                        raytracer_data.materials.emplace_back(
                            raytracer::lambert{math::vec3{rd_real(generator), rd_real(generator), rd_real(generator)}}
                        );
                        break;

                    case 1:
                        raytracer_data.materials.emplace_back(
                            raytracer::metal{math::vec3{rd_real(generator), rd_real(generator), rd_real(generator)}, .5f * rd_real(generator)}
                        );
                        break;

                    case 2:
                        raytracer_data.materials.emplace_back(
                            raytracer::dielectric{math::vec3{rd_real(generator), rd_real(generator), rd_real(generator)}, 1.5f}
                        );
                        break;

                    default:
                        break;
                }
            }
        }
    }
#endif

    raytracer::camera camera{
        math::vec3{-4, 3.2, 5}, math::vec3{0, 1, 0}, math::vec3{0, 1, 0},
        static_cast<float>(app_data.width) / static_cast<float>(app_data.height), 42.f,
        0.0625f, math::distance(math::vec3{-4, 3.2, 5}, math::vec3{0, 1, 0})
    };

    std::vector<math::vec3> multisampling_texels(app_data.sampling_number, math::vec3{0});

    std::vector<math::u8vec3> texels_data(static_cast<std::size_t>(app_data.width) * app_data.height);

    auto random_distribution = std::uniform_real_distribution{0.f, 1.f};

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

            auto color = std::reduce(std::execution::seq, std::begin(multisampling_texels), std::end(multisampling_texels), math::vec3{0});

            color /= static_cast<float>(app_data.sampling_number);

            color = app::gamma_correction(color);

            auto &&rgb = texels_data[x + static_cast<std::size_t>(app_data.width) * y];

            rgb = app::normalize_rgb_to_8bit(std::move(color));
        }
    }

    app::save_to_file("image.ppm"sv, app_data, texels_data);
}

