#include <sstream>

#define CUDA_IMPL
#include "../main.hxx"

#include "math.hxx"


namespace cuda {
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

template<class T>
__device__ math::vec3 color(T &&ray)
{
    return cuda::background_color(ray.unit_direction().y * .5f + .5f);
}
}

__global__
void render(thrust::device_ptr<math::vec3> framebuffer_ptr, std::uint32_t width, std::uint32_t height)
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

    framebuffer_raw_ptr[pixel_index] = cuda::color(ray);
}


void cuda_impl(std::uint32_t width, std::uint32_t height, std::vector<math::u8vec3> &image_texels)
{
    thrust::device_vector<math::vec3> framebuffer(static_cast<std::size_t>(width) * height);

    auto const threads_number = dim3{8, 8, 1};
    auto const blocks_number = dim3{width / threads_number.x + 1, height / threads_number.y + 1, 1};

    render<<<blocks_number, threads_number>>>(framebuffer.data(), width, height);

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);

    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    auto begin = thrust::make_transform_iterator(framebuffer.begin(), cuda::rgb32_to_rgb8{});
    auto end = thrust::make_transform_iterator(framebuffer.end(), cuda::rgb32_to_rgb8{});

    thrust::copy(begin, end, image_texels.begin());
}
