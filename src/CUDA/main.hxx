#pragma once

#include <iostream>
#include <sstream>

#include <string>
using namespace std::string_literals;

#include <vector>
#include <cmath>

#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/count.h>

#include <thrust/iterator/transform_iterator.h>

#include "math.hxx"


namespace cuda {
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
}
