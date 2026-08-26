#include "cuda_utils.cuh"

#include <vector>
#include <iostream>

constexpr int threads_pre_block = 256;
// 一个block有多少thread
constexpr int kTiledThreadsPerBlock = 256;
// 一个thread处理多少元素
constexpr int kItemsPerThread = 4;
// 每个block（tile）处理多少元素
constexpr int kTileSize = kTiledThreadsPerBlock * kItemsPerThread;

// CPU baseline
void vector_add_cpu_baseline(
    const std::vector<float> &a,
    const std::vector<float> &b,
    std::vector<float> &c,
    int count)
{
    for (int i = 0; i < count; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// naive_kernel
__global__ void vector_add_naive_kernel(
    const float *a,
    const float *b,
    float *c,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        c[index] = a[index] + b[index];
    }
}

// grid-stride kernel
// 不是让一个线程负责一个元素，而是以stride去扫数组
__global__ void vector_add_grid_stride_kernel(
    const float *a,
    const float *b,
    float *c,
    int count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < count; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

// vectorized kernel
// 把连续四个float作为一个float4处理
// 使用更宽的数据，优化内存访问
__global__ void vector_add_vectorized_kernel(
    const float4 *a4,
    const float4 *b4,
    float4 *c4,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        const float4 lhs = a4[index];
        const float4 rhs = b4[index];
        c4[index] = make_float4(
            lhs.x + rhs.x,
            lhs.y + rhs.y,
            lhs.z + rhs.z,
            lhs.w + rhs.w);
    }
}

// tiled kernel
__global__ void vector_add_tiled_kernel(
    const float *a,
    const float *b,
    float *c,
    int count)
{
    const int tile_start = blockIdx.x * kTileSize;
    const int thread_base = tile_start + threadIdx.x;
    for (int item = 0; item < kItemsPerThread; item++)
    {
        const int index = thread_base + item * kTiledThreadsPerBlock;
        if (index < count)
        {
            c[index] = a[index] + b[index];
        }
    }
}

void fill_inputs(std::vector<float> &a, std::vector<float> &b)
{
    for (int i = 0; i < static_cast<int>(a.size()); i++)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }
}

void run_naive_vector_add(
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    std::vector<float> &host_c)
{

    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const int count = static_cast<int>(host_a.size());
    const size_t bytes = host_a.size() * sizeof(float);

    // 1. 分配 device memory。
    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));

    // 2. 将两个输入从 host 复制到 device。
    CHECK_CUDA(cudaMemcpy(
        device_a,
        host_a.data(),
        bytes,
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        device_b,
        host_b.data(),
        bytes,
        cudaMemcpyHostToDevice));

    // 3. 计算 kernel launch configuration。
    const int blocks = cuda_utils::ceil_div(count, threads_pre_block);

    // 4. 启动 kernel。
    vector_add_naive_kernel<<<blocks, threads_pre_block>>>(
        device_a,
        device_b,
        device_c,
        count);

    // 检查 kernel launch configuration 等立即错误。
    CHECK_LAST_CUDA_ERROR();

    // 等待 kernel 执行完成，并检查执行阶段错误。
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. 将结果从 device 复制回 host。
    CHECK_CUDA(cudaMemcpy(
        host_c.data(),
        device_c,
        bytes,
        cudaMemcpyDeviceToHost));

    // 6. 释放 device memory。
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
}

void run_grid_stride_vector_add(
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    std::vector<float> &host_c)
{

    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const int count = static_cast<int>(host_a.size());
    const size_t bytes = host_a.size() * sizeof(float);

    // 1. 分配 device memory。
    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));

    // 2. 将两个输入从 host 复制到 device。
    CHECK_CUDA(cudaMemcpy(
        device_a,
        host_a.data(),
        bytes,
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        device_b,
        host_b.data(),
        bytes,
        cudaMemcpyHostToDevice));

    // 3. 计算 kernel launch configuration。
    const int blocks = cuda_utils::ceil_div(count, threads_pre_block);

    // 4. 启动 kernel。
    vector_add_grid_stride_kernel<<<blocks, threads_pre_block>>>(
        device_a,
        device_b,
        device_c,
        count);

    // 检查 kernel launch configuration 等立即错误。
    CHECK_LAST_CUDA_ERROR();

    // 等待 kernel 执行完成，并检查执行阶段错误。
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. 将结果从 device 复制回 host。
    CHECK_CUDA(cudaMemcpy(
        host_c.data(),
        device_c,
        bytes,
        cudaMemcpyDeviceToHost));

    // 6. 释放 device memory。
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
}

void run_vectorized_stride_vector_add(
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    std::vector<float> &host_c)
{

    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const int count = static_cast<int>(host_a.size());
    const size_t bytes = host_a.size() * sizeof(float);

    // 1. 分配 device memory。
    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));

    // 2. 将两个输入从 host 复制到 device。
    CHECK_CUDA(cudaMemcpy(
        device_a,
        host_a.data(),
        bytes,
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        device_b,
        host_b.data(),
        bytes,
        cudaMemcpyHostToDevice));

    // 3. 计算 kernel launch configuration。
    const int vector_count = count / 4;
    const int vectorized_count = vector_count * 4;

    // 4. 启动 kernel。
    if (vector_count > 0)
    {
        const int blocks = cuda_utils::ceil_div(vector_count, threads_pre_block);
        const auto *a4 = reinterpret_cast<const float4 *>(device_a);
        const auto *b4 = reinterpret_cast<const float4 *>(device_b);
        auto *c4 = reinterpret_cast<float4 *>(device_c);
        vector_add_vectorized_kernel<<<blocks, threads_pre_block>>>(a4, b4, c4, vector_count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    if (vectorized_count < count)
    {
        const int tail_count = count - vectorized_count;
        const int tail_blocks = cuda_utils::ceil_div(tail_count, threads_pre_block);
        vector_add_grid_stride_kernel<<<tail_blocks, threads_pre_block>>>(
            device_a + vectorized_count,
            device_b + vectorized_count,
            device_c + vectorized_count,
            tail_count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 5. 将结果从 device 复制回 host。
    CHECK_CUDA(cudaMemcpy(
        host_c.data(),
        device_c,
        bytes,
        cudaMemcpyDeviceToHost));

    // 6. 释放 device memory。
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
}

void run_tiled_vector_add(
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    std::vector<float> &host_c)
{
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const int count = static_cast<int>(host_a.size());
    const size_t bytes = host_a.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));

    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    // tile版本中，grid的单位变成tile数量
    const int blocks = cuda_utils::ceil_div(count, kTileSize);
    vector_add_tiled_kernel<<<blocks, kTiledThreadsPerBlock>>>(device_a, device_b, device_c, count);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
}

bool check_output(
    const std::vector<float> &expected,
    const std::vector<float> &actual)
{
    for (int i = 0; i < static_cast<int>(expected.size()); ++i)
    {
        if (std::fabs(expected[i] - actual[i]) > 1e-5f)
        {
            std::cerr
                << "Correctness: FAIL"
                << ", index=" << i
                << ", expected=" << expected[i]
                << ", actual=" << actual[i]
                << '\n';

            return false;
        }
    }

    std::cout << "Correctness: PASS\n";
    return true;
}

int main()
{
    constexpr int count = 1 << 20;

    // 输入。
    std::vector<float> host_a(count);
    std::vector<float> host_b(count);
    std::vector<float> host_c_naive(count, 0.0f);
    std::vector<float> host_c_tiled(count, 0.0f);
    std::vector<float> host_c_grid_stride(count, 0.0f);
    std::vector<float> host_c_vectorized(count, 0.0f);
    std::vector<float> reference(count, 0.0f);

    fill_inputs(host_a, host_b);
    vector_add_cpu_baseline(host_a, host_b, reference, count);

    // naive版本
    run_naive_vector_add(host_a, host_b, host_c_naive);

    const bool naive_ok = check_output(host_c_naive, reference);

    if (!naive_ok)
    {
        std::cerr << "Naive vector add test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Naive vector add test passed.\n";
    }

    // stride版本
    run_grid_stride_vector_add(host_a, host_b, host_c_grid_stride);

    const bool grid_stride_ok = check_output(host_c_grid_stride, reference);

    if (!grid_stride_ok)
    {
        std::cerr << "Grid-stride vector add test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Grid-stride vector add test passed.\n";
    }

    // vectorizede版本
    run_vectorized_stride_vector_add(host_a, host_b, host_c_vectorized);

    const bool vectorized_ok = check_output(host_c_vectorized, reference);

    if (!vectorized_ok)
    {
        std::cerr << "Vectorized vector add test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Vectorized vector add test passed.\n";
    }

    // tile版本
    run_tiled_vector_add(host_a, host_b, host_c_tiled);

    const bool tiled_ok = check_output(host_c_tiled, reference);

    if (!tiled_ok)
    {
        std::cerr << "Vectorized vector add test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Vectorized vector add test passed.\n";
    }

    return 0;
}