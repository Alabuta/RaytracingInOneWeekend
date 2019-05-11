#include "main.hxx"

struct rgb32_to_rgb8 final {
    __host__ __device__
    RGB<std::uint8_t> operator() (math::vec3 const &color) const noexcept
    {
        return {
            static_cast<std::uint8_t>(color.x * 255.f),
            static_cast<std::uint8_t>(color.y * 255.f),
            static_cast<std::uint8_t>(color.z * 255.f)
        };
    }
};

__global__
void render(thrust::device_ptr<math::vec3> framebuffer_ptr, std::uint32_t width, std::uint32_t height)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(x < width && y < height))
        return;

    auto pixel_index = x + y * width;

    auto framebuffer_raw_ptr = thrust::raw_pointer_cast(framebuffer_ptr);

    framebuffer_raw_ptr[pixel_index] = math::vec3{
        static_cast<float>(x) / width,
        1.f - static_cast<float>(y) / height,
        .2f
    };
}

void cuda_impl(std::uint32_t width, std::uint32_t height, std::vector<RGB<std::uint8_t>> &image_texels)
{
    thrust::device_vector<math::vec3> framebuffer(static_cast<std::size_t>(width) * height);

    auto const threads_number = dim3{8, 8, 1};
    auto const blocks_number = dim3{width / threads_number.x + 1, height / threads_number.y + 1, 1};

    render<<<blocks_number, threads_number>>>(framebuffer.data(), width, height);

    cuda::check_errors(cudaGetLastError(), __FILE__, __LINE__);

    cuda::check_errors(cudaDeviceSynchronize(), __FILE__, __LINE__);

    auto begin = thrust::make_transform_iterator(framebuffer.begin(), rgb32_to_rgb8{});
    auto end = thrust::make_transform_iterator(framebuffer.end(), rgb32_to_rgb8{});

    thrust::copy(begin, end, image_texels.begin());
}
