#include <sstream>

#define CUDA_IMPL
#include "main.hxx"

#include "math.hxx"
#include "primitives.hxx"


namespace cuda {
struct data final {
    static auto constexpr bounces_number = 64u;

    thrust::device_vector<primitives::sphere> spheres;

    thrust::device_ptr<primitives::sphere> spheres_ptr;
    std::uint32_t spheres_size{0};

    //std::vector<material::types> materials;
};

struct rgb32_to_rgb8 final {
    __host__ __device__
    math::u8vec3 operator() (math::vec3 const &color) const noexcept
    {
        return {
            static_cast<std::uint8_t>(color.x * 255.f),
            static_cast<std::uint8_t>(color.y * 255.f),
            static_cast<std::uint8_t>(color.z * 255.f)
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

    auto discriminant =  b * b - a * c;

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
__device__ primitives::hit hit_world(thrust::device_ptr<primitives::sphere> spheres_ptr, std::uint32_t spheres_size, T &&ray)
{
    auto const kMAX = FLT_MAX;// std::numeric_limits<float>::max();
    auto const kMIN = .008f;

    auto spheres_raw_ptr = spheres_ptr.get();//thrust::raw_pointer_cast(spheres_ptr);

    auto min_time = kMAX;

    primitives::hit closest_hit;

    for (auto sphere_index = 0u; sphere_index < spheres_size; ++sphere_index) {
        auto hit = cuda::intersect(ray, spheres_raw_ptr[sphere_index], kMIN, kMAX);

        if (hit.valid) {
            min_time = fmin(min_time, hit.time);

            closest_hit = std::move(hit);
        }
    }

    return closest_hit;
}

template<class T>
__device__ math::vec3 color(thrust::device_ptr<cuda::data> cuda_data, T &&ray)
{
    math::vec3 attenuation{1};

    auto scattered_ray = std::forward<T>(ray);
    //math::vec3 energy_absorption{0};
    
    auto data_raw_ptr = thrust::raw_pointer_cast(cuda_data.get());

    for (auto bounce = 0u; bounce < data_raw_ptr->bounces_number; ++bounce) {
        auto hit = cuda::hit_world(data_raw_ptr->spheres_ptr, data_raw_ptr->spheres_size, scattered_ray);

        if (hit.valid) {
            /*if (auto scattered = raytracer::apply_material(raytracer_data, scattered_ray, hit.value()); scattered) {
                std::tie(scattered_ray, energy_absorption) = *scattered;

                attenuation *= energy_absorption;
            }

            else return math::vec3{0};*/
            return hit.normal * .5f + .5f;
        }

        else return cuda::background_color(scattered_ray.unit_direction().y * .5f + .5f) * attenuation;
    }

    return math::vec3{0};
}

__global__ void populate_world(thrust::device_ptr<cuda::data> data_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto data_raw_ptr = thrust::raw_pointer_cast(data_ptr);

        /*data_raw_ptr->spheres_ptr = thrust::device_malloc<primitives::sphere>(2);

        data_raw_ptr->spheres_ptr[0] = primitives::sphere{math::vec3{0, 1, 0}, 1.f, 0};
        data_raw_ptr->spheres_ptr[1] = primitives::sphere{math::vec3{0, -1000.125f, 0}, 1000.f, 3};*/
    }
}
}

__global__
void render(thrust::device_ptr<cuda::data> cuda_data, thrust::device_ptr<math::vec3> framebuffer_ptr, std::uint32_t width, std::uint32_t height)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(x < width && y < height))
        return;

    auto pixel_index = x + y * width;

    auto framebuffer_raw_ptr = thrust::raw_pointer_cast(framebuffer_ptr);

    auto u = static_cast<float>(x) / width;
    auto v = 1.f - static_cast<float>(y) / height;

    math::vec3 origin{0};

    math::vec3 lower_left_corner{-2, -1, -1};

    math::vec3 horizontal{4, 0, 0};
    math::vec3 vertical{0, 2, 0};

    math::ray ray{origin, lower_left_corner + horizontal * u + vertical * v};

    framebuffer_raw_ptr[pixel_index] = cuda::color(cuda_data, std::move(ray));
}


void cuda_impl(std::uint32_t width, std::uint32_t height, std::vector<math::u8vec3> &image_texels)
{
    thrust::device_vector<math::vec3> framebuffer(static_cast<std::size_t>(width) * height);

    auto const threads_number = dim3{8, 8, 1};
    auto const blocks_number = dim3{width / threads_number.x + 1, height / threads_number.y + 1, 1};

    auto data_ptr = thrust::device_malloc<cuda::data>(1);
    auto data_raw_ptr = thrust::raw_pointer_cast(data_ptr);

    data_raw_ptr->spheres.push_back(primitives::sphere{math::vec3{0, 1, 0}, 1.f, 0});
    data_raw_ptr->spheres.push_back(primitives::sphere{math::vec3{0, -1000.125f, 0}, 1000.f, 3});

    data_raw_ptr->spheres_ptr = thrust::device_ptr<primitives::sphere>{data_raw_ptr->spheres.data()};
    data_raw_ptr->spheres_size = static_cast<std::uint32_t>(data_raw_ptr->spheres.size());

    /*cuda::data cuda_data;

    thrust::device_vector<primitives::sphere> spheres;

    spheres.push_back(primitives::sphere{math::vec3{0, 1, 0}, 1.f, 0});
    spheres.push_back(primitives::sphere{math::vec3{0, -1000.125f, 0}, 1000.f, 3});

    cuda_data.spheres_ptr = thrust::device_ptr<primitives::sphere>{spheres.data()};
    cuda_data.spheres_size = static_cast<std::uint32_t>(spheres.size());*/

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);
    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    render<<<blocks_number, threads_number>>>(data_ptr, framebuffer.data(), width, height);

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);
    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    auto begin = thrust::make_transform_iterator(framebuffer.begin(), cuda::rgb32_to_rgb8{});
    auto end = thrust::make_transform_iterator(framebuffer.end(), cuda::rgb32_to_rgb8{});

    thrust::copy(begin, end, image_texels.begin());

    thrust::device_free(data_ptr);
}
