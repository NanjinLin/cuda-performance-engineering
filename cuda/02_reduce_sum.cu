#include "cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kChunkItemsPerThread = 4;

struct ReductionRun
{
    float value = 0.0f;
    int stages = 0;
};

// cpu-baseline
double reduce_sum_cpu(
    const std::vector<float> &values)
{
    double total = 0.0;
    for (float value : values)
    {
        total += static_cast<double>(value);
    }
    return total;
}

// atomic
// 缺点在于contention很重，性能差
__global__ void reduce_sum_atomic_kernel(
    const float *input,
    float *output,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < count; i += stride)
    {
        atomicAdd(output, input[i]);
    }
}

// shared memory
// 每个block产生sum，对这些sum规约直至只剩一个值
// CUDA reduction思路：先block协作，再多阶段缩小问题规模
__global__ void reduce_sum_shared_kernel(
    const float *input,
    float *block_sums,
    int count)
{
    extern __shared__ float shared[];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    shared[threadIdx.x] = (index < count) ? input[index] : 0.0f;
    __syncthreads();
    // 树形规约 前128吃掉后128， 前64吃掉后64
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        block_sums[blockIdx.x] = shared[0];
    }
}

// warp规约helper
// shuffle可以直接交换寄存器的值而不需要shared memory
__device__ float warp_reduce_sum(float value)

{
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}
// warp + block reduction
// 一个block输出一个sum，但 block内部使用warp
// 步骤：每个warp做小规约，把lane 0的结果放入shared memory，
// 第一个warp再把warp sum规约乘block sum
__global__ void reduce_sum_warp_kernel(
    const float *input,
    float *block_sums,
    int count)
{
    __shared__ float warp_sums[kThreadsPerBlock / kWarpSize];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (index < count) ? input[index] : 0.0f;
    value = warp_reduce_sum(value);
    const int lane = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    if (lane == 0)
    {
        warp_sums[warp_id] = value;
    }
    __syncthreads();
    // 把warp_sum规约成block_sum
    if (warp_id == 0)
    {
        value = (lane < (blockDim.x / kWarpSize)) ? warp_sums[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0)
        {
            block_sums[blockIdx.x] = value;
        }
    }
}

// multi-elements-per-thread
// 每个thread处理多个值
// 在kernel中很常见，以减少“线程管理成本”
__global__ void reduce_sum_chunked_kernel(
    const float *input,
    float *block_sums,
    int count)
{
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * blockDim.x * kChunkItemsPerThread;
    const int thread_start = block_start + tid;
    float local_sum = 0.0f;
    for (int item = 0; item < kChunkItemsPerThread; item++)
    {
        const int index = thread_start + blockDim.x * item;
        if (index < count)
        {
            local_sum += input[index];
        }
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        block_sums[blockIdx.x] = shared[0];
    }
}

void fill_input(std::vector<float> &values)
{
    for (int i = 0; i < static_cast<int>(values.size()); i++)
    {
        values[i] = static_cast<float>(i % 7);
    }
}

bool check_output(float got, double expected, const char *label)
{
    if (std::fabs(static_cast<double>(got) - expected) > 1e-3)
    {
        std::cerr << label << "mismatch:got " << got
                  << ", expected" << expected << "\n";
        return false;
    }
    return true;
}

ReductionRun run_atomic_reduction(const std::vector<float> &host_input)
{
    float *device_input = nullptr;
    float *device_output = nullptr;

    const int count = static_cast<int>(host_input.size());
    const size_t input_bytes = host_input.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&device_output, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));

    const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
    reduce_sum_atomic_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, count);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    ReductionRun result;
    result.stages = 1;
    CHECK_CUDA(cudaMemcpy(&result.value, device_output, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));
    return result;
}

template <typename KernelLauncher>
ReductionRun run_multi_stage_reduction(
    const std::vector<float> &host_input,
    KernelLauncher launch_kernel)
{
    float *current_input = nullptr;
    const size_t input_bytes = host_input.size() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&current_input, input_bytes));
    CHECK_CUDA(cudaMemcpy(current_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));

    int current_count = static_cast<int>(host_input.size());
    int stages = 0;

    // 这里是 reduction 很关键的思想:
    // 一次 kernel launch 并不会直接把 N 个元素变成 1 个元素，
    // 它通常只是把 N 个元素缩成 "每个 block 一个 partial sum"。
    //
    // 所以我们要多次 launch:
    // N -> num_blocks
    // num_blocks -> 更小的 num_blocks
    // ...
    // 直到只剩一个值。
    while (current_count > 1)
    {
        const int blocks = cuda_utils::ceil_div(current_count, kThreadsPerBlock);
        float *next_output = nullptr;
        CHECK_CUDA(cudaMalloc(&next_output, blocks * sizeof(float)));

        launch_kernel(current_input, next_output, current_count, blocks);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaFree(current_input));
        current_input = next_output;
        current_count = blocks;
        ++stages;
    }

    ReductionRun result;
    result.stages = stages;
    CHECK_CUDA(cudaMemcpy(&result.value, current_input, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(current_input));
    return result;
}

ReductionRun run_shared_reduction(const std::vector<float> &host_input)
{
    return run_multi_stage_reduction(
        host_input,
        [](const float *input, float *output, int count, int blocks)
        {
            reduce_sum_shared_kernel<<<blocks, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(
                input, output, count);
        });
}

ReductionRun run_warp_reduction(const std::vector<float> &host_input)
{
    return run_multi_stage_reduction(
        host_input,
        [](const float *input, float *output, int count, int blocks)
        {
            reduce_sum_warp_kernel<<<blocks, kThreadsPerBlock>>>(input, output, count);
        });
}

ReductionRun run_chunked_hierarchical_reduction(const std::vector<float> &host_input)
{
    float *current_input = nullptr;
    const size_t input_bytes = host_input.size() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&current_input, input_bytes));
    CHECK_CUDA(cudaMemcpy(current_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));

    int current_count = static_cast<int>(host_input.size());
    int stages = 0;

    while (current_count > 1)
    {
        const int block_span = kThreadsPerBlock * kChunkItemsPerThread;
        const int blocks = cuda_utils::ceil_div(current_count, block_span);
        float *next_output = nullptr;
        CHECK_CUDA(cudaMalloc(&next_output, blocks * sizeof(float)));

        reduce_sum_chunked_kernel<<<blocks, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(
            current_input, next_output, current_count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaFree(current_input));
        current_input = next_output;
        current_count = blocks;
        ++stages;
    }

    ReductionRun result;
    result.stages = stages;
    CHECK_CUDA(cudaMemcpy(&result.value, current_input, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(current_input));
    return result;
}

int main()
{
    // 故意不选 2 的整次幂，方便你观察:
    // 真正的 kernel 通常要处理各种“尾巴”。
    constexpr int count = (1 << 20) + 37;

    std::vector<float> host_input(count);
    fill_input(host_input);

    // cpu_baseline
    const double expected = reduce_sum_cpu(host_input);

    // atomic
    const ReductionRun atomic_result = run_atomic_reduction(host_input);

    // shared memory
    const ReductionRun shared_result = run_shared_reduction(host_input);

    // warp
    const ReductionRun warp_result = run_warp_reduction(host_input);

    // chunk
    const ReductionRun chunked_result = run_chunked_hierarchical_reduction(host_input);

    const bool atomic_ok = check_output(atomic_result.value, expected, "atomic");
    const bool shared_ok = check_output(shared_result.value, expected, "shared");
    const bool warp_ok = check_output(warp_result.value, expected, "warp");
    const bool chunked_ok = check_output(chunked_result.value, expected, "chunked");

    if (!atomic_ok)
    {
        std::cerr << "atomic test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "atomic test passed.\n";
    }

    if (!shared_ok)
    {
        std::cerr << "shared test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "shared test passed.\n";
    }

    if (!warp_ok)
    {
        std::cerr << "warp test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "warp test passed.\n";
    }

    if (!chunked_ok)
    {
        std::cerr << "chunked test failed.\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "chunked test passed.\n";
    }

    return 0;
}