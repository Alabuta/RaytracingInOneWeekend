#include <sstream>

#define CUDA_IMPL
#include "main.hxx"

#include "math.hxx"
#include "primitives.hxx"
#include "camera.hxx"
#include "material.hxx"


namespace cuda {
struct random_engine final {
    //curandStateXORWOW_t random_state;
    //curandStatePhilox4_32_10_t random_state;
    std::uint32_t random_state;

    random_engine() = default;

    __device__ std::uint32_t rand_xorshift()
    {
        // Xorshift algorithm from George Marsaglia's paper
        random_state ^= (random_state << 13);
        random_state ^= (random_state >> 17);
        random_state ^= (random_state << 5);

        return random_state;
    }

    __device__ random_engine(std::uint32_t seed)
    {
        //curand_init(1234, seed, 0, &random_state);
        random_state = seed;
    }

    __device__ float generate()
    {
        //return curand_uniform(&random_state);
        return static_cast<float>(rand_xorshift()) * (1.f / 4294967296.f);
    }

    __device__ math::vec3 random_in_unit_sphere()
    {
        math::vec3 vector;

        do {
            vector = math::vec3{
                generate() * 2.f - 1.f,
                generate() * 2.f - 1.f,
                generate() * 2.f - 1.f
            };
        } while (math::length(vector) > 1.f);

        return vector;
    }
};

struct data final {
    static auto constexpr sampling_number = 128u;
    static auto constexpr bounces_number = 64u;

    std::uint32_t width{0}, height{0};

    cuda::random_engine *random_engines;

    raytracer::camera camera;

    primitives::sphere *spheres_array;
    std::uint32_t spheres_size{0};

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

__device__ float schlick_reflection_probability(float refraction_index, float cosine_theta)
{
    auto reflection_coefficient = std::pow((1.f - refraction_index) / (1.f + refraction_index), 2);

    return reflection_coefficient + (1.f - reflection_coefficient) * std::pow(1.f - cosine_theta, 5);
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
__device__ primitives::hit hit_world(cuda::data &cuda_data, T &&ray)
{
    auto const kMAX = FLT_MAX;// std::numeric_limits<float>::max();
    auto const kMIN = .008f;

    auto min_time = kMAX;

    primitives::hit closest_hit;

    for (auto sphere_index = 0u; sphere_index < cuda_data.spheres_size; ++sphere_index) {
        auto hit = cuda::intersect(ray, cuda_data.spheres_array[sphere_index], kMIN, min_time);

        if (hit.valid) {
            min_time = hit.time;
            closest_hit = std::move(hit);
        }
    }

    return closest_hit;
}

struct apply_material final {
    cuda::random_engine &random_engine;
    math::ray ray;
    primitives::hit hit;

    __device__
    apply_material(cuda::random_engine &random_engine, math::ray const &ray, primitives::hit const &hit)
        : random_engine{random_engine}, ray{ray}, hit{hit} { }

    __device__
    material::surface_response operator() (material::lambert const &material)
    {
        auto random_direction = math::normalize(random_engine.random_in_unit_sphere());
        auto target = hit.normal + random_direction;

        auto scattered_ray = math::ray{hit.position, target};
        auto attenuation = material.albedo;

        return material::surface_response{scattered_ray, attenuation, true};
    }

    __device__
    material::surface_response operator() (material::metal const &material)
    {
        auto reflected = math::reflect(ray.unit_direction(), hit.normal);

        auto random_direction = math::normalize(random_engine.random_in_unit_sphere());

        auto scattered_ray = math::ray{hit.position, reflected + random_direction * material.roughness};
        auto attenuation = material.albedo;

        if (math::dot(scattered_ray.direction, hit.normal) > 0.f)
            return material::surface_response{scattered_ray, attenuation, true};

        return { };
    }

    __device__
    material::surface_response operator() (material::dielectric const &material)
    {
        auto outward_normal = -hit.normal;
        auto refraction_index = material.refraction_index;
        auto cosine_theta = math::dot(ray.unit_direction(), hit.normal);

        if (cosine_theta <= 0.f) {
            outward_normal *= -1.f;
            refraction_index = 1.f / refraction_index;
            cosine_theta *= -1.f;
        }

        auto attenuation = material.albedo;
        auto refracted = math::refract(ray.unit_direction(), outward_normal, refraction_index);

        auto reflection_probability = 1.f;

        if (math::length(refracted) > 0.f)
            reflection_probability = cuda::schlick_reflection_probability(refraction_index, cosine_theta);

        math::ray scattered_ray;

        if (random_engine.generate() < reflection_probability) {
            auto reflected = math::reflect(ray.unit_direction(), hit.normal);
            scattered_ray = math::ray{hit.position, reflected};
        }

        else scattered_ray = math::ray{hit.position, refracted};

        return material::surface_response{scattered_ray, attenuation, true};
    }
};

template<class T>
__device__ math::vec3 color(cuda::data &cuda_data, cuda::random_engine &random_engine, T &&ray)
{
    math::vec3 attenuation{1};

    auto scattered_ray = std::forward<T>(ray);
    //math::vec3 energy_absorption{0};
    
    for (auto bounce = 0u; bounce < cuda_data.bounces_number; ++bounce) {
        auto hit = cuda::hit_world(cuda_data, scattered_ray);

        if (hit.valid) {
            auto const material = raytracer_data.materials[hit.material_index];

            auto surface_response = variant::apply_visitor(cuda::apply_material(random_engine, scattered_ray, hit), material);

            if (surface_response.valid) {
                scattered_ray = surface_response.ray;
                energy_absorption = surface_response.attenuation;

                attenuation *= energy_absorption;
            }

            else return math::vec3{0};
        }

        else return cuda::background_color(scattered_ray.unit_direction().y * .5f + .5f) * attenuation;
    }

    return math::vec3{0};
}

__global__ void init_raytracer_data(
    thrust::device_ptr<cuda::data> data_ptr,
    std::uint32_t width, std::uint32_t height,
    thrust::device_ptr<primitives::sphere> spheres_ptr, std::uint32_t spheres_size,
    thrust::device_ptr<material::types> materials_ptr,
    thrust::device_ptr<cuda::random_engine> random_engines)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    auto data_raw_ptr = thrust::raw_pointer_cast(data_ptr);

    data_raw_ptr->width = width;
    data_raw_ptr->height = height;

    data_raw_ptr->random_engines = thrust::raw_pointer_cast(random_engines);

    data_raw_ptr->camera = raytracer::camera{
        math::vec3{0, 0, 0}, math::vec3{0, 0, -1}, math::vec3{0, 1, 0},
        static_cast<float>(width) / static_cast<float>(height), 90.f,
        .0625f, 1.f//math::distance(math::vec3{-4, 3.2, 5}, math::vec3{0, 1, 0})
    };

    data_raw_ptr->spheres_size = spheres_size;
    data_raw_ptr->spheres_array = thrust::raw_pointer_cast(spheres_ptr);
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

    auto &random_engine = data_raw_ptr->random_engines[pixel_index];

    math::vec3 color{0};

    for (auto s = 0u; s < data_raw_ptr->sampling_number; ++s) {
        auto u = static_cast<float>(x + random_engine.generate()) / width;
        auto v = static_cast<float>(y + random_engine.generate()) / height;

        color += cuda::color(*data_raw_ptr, random_engine, data_raw_ptr->camera.ray(u, v));
    }

    color /= static_cast<float>(data_raw_ptr->sampling_number);

    framebuffer_ptr.get()[pixel_index] = color;
}
}


void cuda_impl(std::uint32_t width, std::uint32_t height, std::vector<math::u8vec3> &image_texels)
{
    auto const threads_number = dim3{8, 8, 1};
    auto const blocks_number = dim3{width / threads_number.x + 1, height / threads_number.y + 1, 1};

    auto const pixels_number = static_cast<std::size_t>(width) * height;

    std::cout << "pixels_number "s << pixels_number << '\n';

    thrust::device_vector<cuda::random_engine> random_engines(pixels_number);

    {
        thrust::device_vector<std::uint64_t> pixel_indices(pixels_number);
        thrust::sequence(thrust::device, pixel_indices.begin(), pixel_indices.end(), 0);

        thrust::transform(thrust::device, pixel_indices.begin(), pixel_indices.end(), random_engines.begin(),
                          [] __device__(std::size_t pixel_index)
        {
            return cuda::random_engine(pixel_index);
        });
    }
    thrust::device_vector<material::types> materials;

    materials.push_back(material::lambert{math::vec3{.1, .2, .5}});
    materials.push_back(material::metal{math::vec3{.8, .6, .2}, 0});
    materials.push_back(material::dielectric{math::vec3{1}, 1.5f});
    materials.push_back(material::lambert{math::vec3{.64, .8, .0}});

    thrust::device_vector<primitives::sphere> spheres;

    spheres.push_back(primitives::sphere{math::vec3{0, 0, -1}, .5f, 0});
    spheres.push_back(primitives::sphere{math::vec3{0, -100.5f, -1}, 100.f, 3});
    spheres.push_back(primitives::sphere{math::vec3{+1, .5f, 0}, .5f, 1});
    spheres.push_back(primitives::sphere{math::vec3{-1, .5f, 0}, .5f, 2});
    spheres.push_back(primitives::sphere{math::vec3{-1, .5f, 0}, -.499f, 2});

    auto data_ptr = thrust::device_malloc<cuda::data>(1);

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);
    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    cuda::init_raytracer_data<<<1, 1>>>(
        data_ptr,
        width, height,
        spheres.data(), static_cast<std::uint32_t>(spheres.size()),
        materials.data(),
        random_engines.data()
    );

    thrust::device_vector<math::vec3> framebuffer(pixels_number);

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);
    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    cuda::render<<<blocks_number, threads_number>>>(data_ptr, framebuffer.data());

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);
    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    thrust::device_free(data_ptr);

    auto begin = thrust::make_transform_iterator(framebuffer.begin(), cuda::rgb32_to_rgb8{});
    auto end = thrust::make_transform_iterator(framebuffer.end(), cuda::rgb32_to_rgb8{});

    thrust::copy(begin, end, image_texels.begin());
}
