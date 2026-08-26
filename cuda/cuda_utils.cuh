#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace cuda_utils
{

  inline void check(cudaError_t status, const char *operation)
  {
    if (status != cudaSuccess)
    {
      std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  inline int ceil_div(int value, int divisor)
  {
    return (value + divisor - 1) / divisor;
  }

} // namespace cuda_utils

#define CHECK_CUDA(expr) ::cuda_utils::check((expr), #expr)
#define CHECK_LAST_CUDA_ERROR() ::cuda_utils::check(cudaGetLastError(), "cudaGetLastError()")