#include <iostream>
#include <vector>
#include <cmath>

#include <string>
using namespace std::string_literals;

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/count.h>

__global__
void add(unsigned int N, thrust::device_ptr<float> a, thrust::device_ptr<float> b)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;

    auto a_raw_ptr = thrust::raw_pointer_cast(a);
    auto b_raw_ptr = thrust::raw_pointer_cast(b);

    for (auto i = index; i < N; i += stride)
        a_raw_ptr[i] += b_raw_ptr[i];
}

void gpgpuWrapper()
{
    auto constexpr N = 1'048'576u;

    auto constexpr kBLOCK_DIM = 256;
    auto constexpr kGRID_DIM = (N + kBLOCK_DIM - 1) / kBLOCK_DIM;

    thrust::device_vector<float> device_a(N, 1.f);
    thrust::device_vector<float> device_b(N, 2.f);

    add<<<kGRID_DIM, kBLOCK_DIM>>>(N, device_a.data(), device_b.data());

    auto error_code = cudaDeviceSynchronize();

    if (error_code != cudaError::cudaSuccess)
        std::cout << "An error is happened: "s << std::string{cudaGetErrorString(error_code)} << '\n';

    auto count = thrust::count(device_a.begin(), device_a.end(), 3.f);

    std::cout << std::boolalpha << "Range is different: "s << (count != N) << '\n';
}
