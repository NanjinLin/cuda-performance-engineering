#include "cuda_utils.cuh"

#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>

constexpr int threads_pre_block = 256;
// 一个block有多少thread
constexpr int kTiledThreadsPerBlock = 256;
// 一个thread处理多少元素
constexpr int kItemsPerThread = 4;
// 每个block（tile）处理多少元素
constexpr int kTileSize = kTiledThreadsPerBlock * kItemsPerThread;
constexpr int num_warmups = 1;
constexpr int num_trials = 3;

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

void launch_naive(
    const float *device_a,
    const float *device_b,
    float *device_c,
    int count)
{
    const int blocks = cuda_utils::ceil_div(count, threads_pre_block);
    vector_add_naive_kernel<<<blocks, threads_pre_block>>>(device_a, device_b, device_c, count);
}

void launch_grid_stride(
    const float *device_a,
    const float *device_b,
    float *device_c,
    int count)
{
    const int blocks = cuda_utils::ceil_div(count, threads_pre_block);
    vector_add_grid_stride_kernel<<<blocks, threads_pre_block>>>(device_a, device_b, device_c, count);
}

void launch_tiled(
    const float *device_a,
    const float *device_b,
    float *device_c,
    int count)
{
    const int blocks = cuda_utils::ceil_div(count, kTileSize);

    vector_add_tiled_kernel<<<blocks, kTiledThreadsPerBlock>>>(device_a, device_b, device_c, count);
}

void launch_float4(
    const float *device_a,
    const float *device_b,
    float *device_c,
    int count)
{
    // 一个 float4 包含 4 个 float。
    const int vector_count = count / 4;
    const int vectorized_count = vector_count * 4;

    // 主体部分：每个 thread 处理一个 float4。
    if (vector_count > 0)
    {
        const int blocks =
            cuda_utils::ceil_div(
                vector_count,
                threads_pre_block);

        const float4 *device_a4 =
            reinterpret_cast<const float4 *>(device_a);

        const float4 *device_b4 =
            reinterpret_cast<const float4 *>(device_b);

        float4 *device_c4 =
            reinterpret_cast<float4 *>(device_c);

        vector_add_vectorized_kernel<<<blocks, threads_pre_block>>>(device_a4, device_b4, device_c4, vector_count);
    }

    // 尾部不足 4 个元素时，用普通 float kernel 处理。
    if (vectorized_count < count)
    {
        const int tail_count = count - vectorized_count;

        const int tail_blocks = cuda_utils::ceil_div(tail_count, threads_pre_block);

        vector_add_grid_stride_kernel<<<tail_blocks, threads_pre_block>>>(device_a + vectorized_count, device_b + vectorized_count, device_c + vectorized_count, tail_count);
    }
}

using VectorAddLaunchFunction = void (*)(
    const float *,
    const float *,
    float *,
    int);
float benchmark_kernel(
    VectorAddLaunchFunction launch,
    const float *device_a,
    const float *device_b,
    float *device_c,
    int count,
    int num_warmups,
    int num_trials)
{
    double total_ms = 0.0;

    // Warmup
    for (int i = 0; i < num_warmups; ++i)
    {
        launch(device_a, device_b, device_c, count);
    }

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 每个 trial 单独测量一次。
    for (int trial = 0; trial < num_trials; trial++)
    {
        CHECK_CUDA(cudaEventRecord(start));

        launch(device_a, device_b, device_c, count);

        // 先立即提交 stop event，避免把额外的 host 工作
        // 插入 kernel 与 stop event 之间。
        CHECK_CUDA(cudaEventRecord(stop));

        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop));

        total_ms += elapsed_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / num_trials;
}

bool check_output(
    const std::vector<float> &actual,
    const std::vector<float> &expected)
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
    constexpr int count = 1 << 24;

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

    // correctness

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
        std::cerr << "Tiled vector add test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "tiled vector add test passed.\n";
    }

    // benchmarking
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const size_t bytes = host_a.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));
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

    // naive
    const float naive_mean_ms =
        benchmark_kernel(
            launch_naive,
            device_a,
            device_b,
            device_c,
            count,
            num_warmups,
            num_trials);

    std::cout << "naive_mean_ms is " << naive_mean_ms << "\n";

    // grid-stride
    const float grid_stride_mean_ms =
        benchmark_kernel(
            launch_grid_stride,
            device_a,
            device_b,
            device_c,
            count,
            num_warmups,
            num_trials);

    std::cout << "grid_stride_mean_ms is " << grid_stride_mean_ms << "\n";

    // tiled
    const float tiled_mean_ms =
        benchmark_kernel(
            launch_tiled,
            device_a,
            device_b,
            device_c,
            count,
            num_warmups,
            num_trials);

    std::cout << "tiled_mean_ms is " << tiled_mean_ms << "\n";

    // float4
    const float float4_mean_ms =
        benchmark_kernel(
            launch_float4,
            device_a,
            device_b,
            device_c,
            count,
            num_warmups,
            num_trials);

    std::cout << "float4_mean_ms is " << float4_mean_ms << "\n";

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    return 0;
}