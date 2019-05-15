#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <chrono>
#include <cmath>

#include <string>
using namespace std::string_literals;

#ifdef CUDA_IMPL
    #include <cuda.h>

    #include <thrust/device_malloc.h>
    #include <thrust/device_free.h>
    #include <curand_kernel.h>

    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>

    #include <thrust/count.h>

    #include <thrust/iterator/transform_iterator.h>
#endif


#ifdef CUDA_IMPL
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST
    #define CUDA_DEVICE
    #define CUDA_HOST_DEVICE
#endif
