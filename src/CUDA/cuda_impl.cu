#include <sstream>

#define CUDA_IMPL
#include "main.hxx"

#include "math.hxx"
#include "primitives.hxx"
#include "camera.hxx"


namespace cuda {
struct data final {
    static auto constexpr sampling_number = 128u;
    static auto constexpr bounces_number = 64u;

    std::uint32_t width{0}, height{0};

    thrust::device_ptr<primitives::sphere> spheres_ptr;
    std::uint32_t spheres_size{0};

    raytracer::camera camera;

    thrust::device_ptr<curandState> random_states_ptr;

    //std::vector<material::types> materials;
};

struct rgb32_to_rgb8 final {
    template<class T>
    __host__ __device__ math::u8vec3 operator() (T &&color) const
    {
        auto const gamma = math::vec3{1.f / 2.2f};

        auto srgb = math::pow(std::forward<T>(color), gamma);

        return {
            static_cast<std::uint8_t>(srgb.x * 255.f),
            static_cast<std::uint8_t>(srgb.y * 255.f),
            static_cast<std::uint8_t>(srgb.z * 255.f)
        };
    }
};

void check_errors(cudaError_t error_code, std::string const &file_name, std::int32_t file_line)
{
    if (error_code == cudaError::cudaSuccess)
        return;

    std::ostringstream error_stream;

    error_stream << "CUDA error "s << std::string{cudaGetErrorString(error_code)} << " at "s;
    error_stream << file_name << "("s << file_line << ")"s << std::endl;

    cudaDeviceReset();

    throw std::runtime_error(error_stream.str());
}

__device__ math::vec3 random_in_unit_sphere(curandState *local_random_state)
{
    math::vec3 vector;

    do {
        vector = math::vec3{
            curand_uniform(local_random_state) * 2.f - 1.f,
            curand_uniform(local_random_state) * 2.f - 1.f,
            curand_uniform(local_random_state) * 2.f - 1.f
        };
    } while (math::length(vector) > 1.f);

    return vector;
}

__device__ math::vec3 background_color(float t)
{
    return math::mix(math::vec3{1}, math::vec3{.5, .7, 1}, t);
}

template<class T1, class T2, typename std::enable_if_t<std::is_same_v<primitives::sphere, std::decay_t<T2>>>* = 0>
__device__ primitives::hit intersect(T1 &&ray, T2 &&sphere, float time_min, float time_max)
{
    auto oc = ray.origin - sphere.center;

    auto a = math::dot(ray.direction, ray.direction);
    auto b = math::dot(oc, ray.direction);
    auto c = math::dot(oc, oc) - sphere.radius * sphere.radius;

    auto discriminant = b * b - a * c;

    if (discriminant > 0.f) {
        float temp = (-b - std::sqrt(b * b - a * c)) / a;

        if (temp < time_max && temp > time_min) {
            auto position = ray.point_at(temp);

            return primitives::hit{
                position,
                (position - sphere.center) / sphere.radius,
                temp,
                sphere.material_index,
                true
            };
        }

        temp = (-b + std::sqrt(b * b - a * c)) / a;

        if (temp < time_max && temp > time_min) {
            auto position = ray.point_at(temp);

            return primitives::hit{
                position,
                (position - sphere.center) / sphere.radius,
                temp,
                sphere.material_index,
                true
            };
        }
    }

    return { };
}

template<class T>
__device__ primitives::hit hit_world(thrust::device_ptr<cuda::data> cuda_data, T &&ray)
{
    auto const kMAX = FLT_MAX;// std::numeric_limits<float>::max();
    auto const kMIN = .008f;

    auto min_time = kMAX;

    auto data_raw_ptr = thrust::raw_pointer_cast(cuda_data);

    auto spheres_raw_ptr = thrust::raw_pointer_cast(data_raw_ptr->spheres_ptr);

    primitives::hit closest_hit;

    for (auto sphere_index = 0u; sphere_index < data_raw_ptr->spheres_size; ++sphere_index) {
        auto hit = cuda::intersect(ray, spheres_raw_ptr[sphere_index], kMIN, min_time);

        if (hit.valid) {
            min_time = hit.time;
            closest_hit = std::move(hit);
        }
    }

    return closest_hit;
}

template<class T>
__device__ math::vec3 color(thrust::device_ptr<cuda::data> cuda_data, curandState *local_random_state, T &&ray)
{
    math::vec3 attenuation{1};

    auto scattered_ray = std::forward<T>(ray);
    //math::vec3 energy_absorption{0};
    
    auto data_raw_ptr = thrust::raw_pointer_cast(cuda_data);

    for (auto bounce = 0u; bounce < data_raw_ptr->bounces_number; ++bounce) {
        auto hit = cuda::hit_world(cuda_data, scattered_ray);

        if (hit.valid) {
            /*if (auto scattered = raytracer::apply_material(raytracer_data, scattered_ray, hit.value()); scattered) {
                std::tie(scattered_ray, energy_absorption) = *scattered;

                attenuation *= energy_absorption;
            }

            else return math::vec3{0};*/

            auto random_direction = cuda::random_in_unit_sphere(local_random_state);
            auto target = hit.position + hit.normal + random_direction;

            scattered_ray = math::ray{hit.position, target - hit.position};

            attenuation *= .5f;
        }

        else return cuda::background_color(scattered_ray.unit_direction().y * .5f + .5f) * attenuation;
    }

    return math::vec3{0};
}

__global__ void init_raytracer_data(
    thrust::device_ptr<cuda::data> data_ptr,
    std::uint32_t width, std::uint32_t height,
    thrust::device_ptr<primitives::sphere> spheres_ptr, std::uint32_t spheres_size,
    thrust::device_ptr<curandState> random_states_ptr)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    auto data_raw_ptr = thrust::raw_pointer_cast(data_ptr);

    data_raw_ptr->width = width;
    data_raw_ptr->height = height;

    data_raw_ptr->spheres_size = spheres_size;
    data_raw_ptr->spheres_ptr = spheres_ptr;
    data_raw_ptr->random_engines = thrust::raw_pointer_cast(random_engines);

    data_raw_ptr->camera = raytracer::camera{
        math::vec3{0, 0, 0}, math::vec3{0, 0, -1}, math::vec3{0, 1, 0},
        static_cast<float>(width) / static_cast<float>(height), 42.f,
        .0625f, 1.f//math::distance(math::vec3{-4, 3.2, 5}, math::vec3{0, 1, 0})
    };

    data_raw_ptr->random_states_ptr = random_states_ptr;
}
}

__global__
void render(thrust::device_ptr<cuda::data> cuda_data, thrust::device_ptr<math::vec3> framebuffer_ptr)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    auto data_raw_ptr = thrust::raw_pointer_cast(cuda_data);

    auto const width = data_raw_ptr->width;
    auto const height = data_raw_ptr->height;

    if (!(x < width && y < height))
        return;

    auto pixel_index = x + y * width;

    auto framebuffer_raw_ptr = thrust::raw_pointer_cast(framebuffer_ptr);

    auto u = static_cast<float>(x) / width;
    auto v = 1.f - static_cast<float>(y) / height;

    auto random_states_raw_ptr = thrust::raw_pointer_cast(data_raw_ptr->random_states_ptr);
    auto local_random_state = random_states_raw_ptr[pixel_index];

    math::vec3 color{0};

    for (auto s = 0u; s < data_raw_ptr->sampling_number; ++s) {
        auto _u = u + curand_uniform(&local_random_state) / static_cast<float>(width);
        auto _v = v + curand_uniform(&local_random_state) / static_cast<float>(height);

        math::ray ray{origin, lower_left_corner + horizontal * _u + vertical * _v};

        color += cuda::color(*data_raw_ptr, random_engine, data_raw_ptr->camera.ray(u, v));
    }

    color /= static_cast<float>(data_raw_ptr->sampling_number);

    framebuffer_raw_ptr[pixel_index] = color;
}


void cuda_impl(std::uint32_t width, std::uint32_t height, std::vector<math::u8vec3> &image_texels)
{
    auto const threads_number = dim3{8, 8, 1};
    auto const blocks_number = dim3{width / threads_number.x + 1, height / threads_number.y + 1, 1};

    auto const pixels_number = static_cast<std::size_t>(width) * height;

    thrust::device_vector<curandState> random_states(pixels_number);

    {
        thrust::device_vector<std::size_t> pixel_indices(pixels_number);
        thrust::sequence(thrust::device, pixel_indices.begin(), pixel_indices.end(), 0);

        thrust::transform(thrust::device, pixel_indices.begin(), pixel_indices.end(), random_states.begin(),
                          [] __device__ (std::size_t pixel_index)
        {
            curandState random_state;

            curand_init(1984, pixel_index, 0, &random_state);

            return random_state;
        });
    }

    auto data_ptr = thrust::device_malloc<cuda::data>(1);

    thrust::device_vector<primitives::sphere> spheres;

    spheres.push_back(primitives::sphere{math::vec3{0, 0, -1}, .5f, 0});
    spheres.push_back(primitives::sphere{math::vec3{0, -100.5f, -1}, 100.f, 3});

    cuda::init_raytracer_data<<<1, 1>>>(
        data_ptr,
        width, height,
        spheres.data(), static_cast<std::uint32_t>(spheres.size()),
        random_states.data()
    );

    thrust::device_vector<math::vec3> framebuffer(pixels_number);

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);
    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    render<<<blocks_number, threads_number>>>(data_ptr, framebuffer.data());

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);
    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    auto begin = thrust::make_transform_iterator(framebuffer.begin(), cuda::rgb32_to_rgb8{});
    auto end = thrust::make_transform_iterator(framebuffer.end(), cuda::rgb32_to_rgb8{});

    thrust::copy(begin, end, image_texels.begin());

    thrust::device_free(data_ptr);
}
