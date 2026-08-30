#include "cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

constexpr int kThreadsPerBlock = 256;
const float kNegInf = -std::numeric_limits<float>::infinity();
struct SoftmaxRun
{
    int rows = 0;
    int cols = 0;
};

void softmax_cpu(
    const std::vector<float> &input,
    std::vector<float> &output,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; row++)
    {
        const float *row_ptr = input.data() + row * cols;
        float *out_ptr = output.data() + row * cols;

        float max_value = row_ptr[0];
        for (int col = 1; col < cols; col++)
        {
            max_value = std::max(max_value, row_ptr[col]);
        }
        double sum = 0.0;
        for (int col = 0; col < cols; col++)
        {
            sum += std::exp(static_cast<double>(row_ptr[col] - max_value));
        }
        for (int col = 0; col < cols; col++)
        {
            out_ptr[col] = static_cast<float>(std::exp(static_cast<double>(row_ptr[col] - max_value)) / sum);
        }
    }
}

void softmax_onlie_cpu(
    const std::vector<float> &input,
    std::vector<float> &output,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; row++)
    {
        const float *row_ptr = input.data() + row * cols;
        float *out_ptr = output.data() + row * cols;

        float row_max = kNegInf;
        double row_sum = 0.0;
        for (int col = 0; col < cols; col++)
        {
            const float x = row_ptr[col];
            const float new_row_max = std::max(row_max, x);
            row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) + std::exp(static_cast<double>(x - new_row_max));
            row_max = new_row_max;
        }
        for (int col = 0; col < cols; col++)
        {
            out_ptr[col] = static_cast<float>(std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
        }
    }
}

void softmax_masked_cpu(
    const std::vector<float> &input,
    const std::vector<int> &mask,
    std::vector<float> &output,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; row++)
    {
        const float *row_ptr = input.data() + row * cols;
        const int *mask_ptr = mask.data() + row * cols;
        float *out_ptr = output.data() + row * cols;

        float row_max = kNegInf;
        double row_sum = 0.0;
        for (int col = 0; col < cols; col++)
        {
            if (mask_ptr[col] == 0)
            {
                continue;
            }
            const float x = row_ptr[col];
            const float new_row_max = std::max(row_max, x);
            row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) + std::exp(static_cast<double>(x - new_row_max));
            row_max = new_row_max;
        }

        for (int col = 0; col < cols; col++)
        {
            if (mask_ptr[col] == 0)
            {
                out_ptr[col] = 0.0f;
            }
            else
            {
                out_ptr[col] = static_cast<float>(std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
            }
        }
    }
}

void softmax_causal_cpu(
    const std::vector<float> &input,
    std::vector<float> &output,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; ++row)
    {
        const float *row_ptr = input.data() + row * cols;
        float *out_ptr = output.data() + row * cols;

        float row_max = kNegInf;
        double row_sum = 0.0;
        for (int col = 0; col <= row && col < cols; ++col)
        {
            const float x = row_ptr[col];
            const float new_row_max = std::max(row_max, x);
            row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
                      std::exp(static_cast<double>(x - new_row_max));
            row_max = new_row_max;
        }

        for (int col = 0; col < cols; ++col)
        {
            if (col > row)
            {
                out_ptr[col] = 0.0f;
            }
            else
            {
                out_ptr[col] = static_cast<float>(
                    std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
            }
        }
    }
}

// naive版本
// 一行只用一个thread,GPU并行度很差
__global__ void softmax_naive_kernel(
    const float *input,
    float *output,
    int rows,
    int cols)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    const float *row_ptr = input + row * cols;
    float *out_ptr = output + row * cols;

    float max_value = row_ptr[0];
    for (int col = 1; col < cols; col++)
    {
        max_value = fmaxf(max_value, row_ptr[col]);
    }
    // benchmark时改为float
    float sum = 0.0f;
    for (int col = 0; col < cols; col++)
    {
        sum += expf(row_ptr[col] - max_value);
    }

    for (int col = 0; col < cols; col++)
    {
        out_ptr[col] = expf(row_ptr[col] - max_value) / sum;
    }
}

// online gpu
__global__ void softmax_online_kernel(
    const float *input,
    float *output,
    int rows,
    int cols)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }
    const float *row_ptr = input + row * cols;
    float *out_ptr = output + row * cols;

    float row_max = kNegInf;
    // benchmark时改为float
    float row_sum = 0.0f;
    for (int col = 0; col < cols; col++)
    {
        const float x = row_ptr[col];
        const float new_row_max = fmaxf(row_max, x);
        row_sum = row_sum * expf(row_max - new_row_max) + expf(x - new_row_max);
        row_max = new_row_max;
    }

    for (int col = 0; col < cols; col++)
    {
        out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
    }
}

// masked softmax
__global__ void softmax_masked_kernel(
    const float *input,
    const int *mask,
    float *output,
    int rows,
    int cols)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    const float *row_ptr = input + row * cols;
    const int *mask_ptr = mask + row * cols;
    float *out_ptr = output + row * cols;

    float row_max = kNegInf;
    double row_sum = 0.0;
    for (int col = 0; col < cols; ++col)
    {
        if (mask_ptr[col] == 0)
        {
            continue;
        }
        const float x = row_ptr[col];
        const float new_row_max = fmaxf(row_max, x);
        row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
                  std::exp(static_cast<double>(x - new_row_max));
        row_max = new_row_max;
    }

    for (int col = 0; col < cols; ++col)
    {
        if (mask_ptr[col] == 0)
        {
            out_ptr[col] = 0.0f;
        }
        else
        {
            out_ptr[col] = static_cast<float>(
                std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
        }
    }
}

// casual softmax
__global__ void softmax_causal_kernel(
    const float *input,
    float *output,
    int rows,
    int cols)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    const float *row_ptr = input + row * cols;
    float *out_ptr = output + row * cols;

    float row_max = kNegInf;
    double row_sum = 0.0;
    for (int col = 0; col <= row && col < cols; ++col)
    {
        const float x = row_ptr[col];
        const float new_row_max = fmaxf(row_max, x);
        row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
                  std::exp(static_cast<double>(x - new_row_max));
        row_max = new_row_max;
    }

    for (int col = 0; col < cols; ++col)
    {
        if (col > row)
        {
            out_ptr[col] = 0.0f;
        }
        else
        {
            out_ptr[col] = static_cast<float>(
                std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
        }
    }
}

// block
__global__ void softmax_block_kernel(
    const float *input,
    float *output,
    int rows,
    int cols)
{
    __shared__ float shared_max[kThreadsPerBlock];
    __shared__ float shared_sum[kThreadsPerBlock];

    const int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    const float *row_ptr = input + row * cols;
    float *out_ptr = output + row * cols;

    // 每个 thread 先扫自己负责的一部分列。
    float local_max = kNegInf;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        local_max = fmaxf(local_max, row_ptr[col]);
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    // 第一次规约: 找整行的最大值。
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    const float row_max = shared_max[0];

    // 第二步: 每个 thread 计算自己负责位置的 exp sum 部分。
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        local_sum += expf(row_ptr[col] - row_max);
    }
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // 第二次规约: 求整行的 sum。
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    const float row_sum = shared_sum[0];

    // 第三步: 归一化写回。
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
    }
}

// block + online 理论性能王者
__global__ void softmax_block_online_kernel(
    const float *input,
    float *output,
    int rows,
    int cols)
{
    __shared__ float shared_max[kThreadsPerBlock];
    __shared__ float shared_sum[kThreadsPerBlock];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows)
    {
        return;
    }

    const float *row_ptr = input + row * cols;
    float *out_ptr = output + row * cols;

    float local_max = kNegInf;
    float local_sum = 0.0f;

    // 每个线程遍历 col = tid, tid + blockDim.x, ...
    // 使用 online 公式更新 local_max 和 local_sum。
    for (int col = tid; col < cols; col += blockDim.x)
    {
        const float x = row_ptr[col];
        const float new_local_max = fmaxf(local_max, x);
        local_sum = local_sum * expf(local_max - new_local_max) + expf(x - new_local_max);
        local_max = new_local_max;
    }

    shared_max[tid] = local_max;
    shared_sum[tid] = local_sum;
    __syncthreads();

    // 对 (shared_max, shared_sum) 进行联合 reduction。
    // 合并左右两个状态时使用上面的公式。
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            const float left_max = shared_max[tid];
            const float left_sum = shared_sum[tid];

            const float right_max = shared_max[tid + offset];
            const float right_sum = shared_sum[tid + offset];

            const float new_max = fmaxf(left_max, right_max);

            const float new_sum =
                left_sum * expf(left_max - new_max) +
                right_sum * expf(right_max - new_max);

            shared_max[tid] = new_max;
            shared_sum[tid] = new_sum;
        }
        __syncthreads();
    }

    const float row_max = shared_max[0];
    const float row_sum = shared_sum[0];

    // 每个线程负责一部分列，完成归一化写回。
    for (int col = tid; col < cols; col += blockDim.x)
    {
        out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
    }
}

void fill_input(std::vector<float> &input, int rows, int cols)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            input[row * cols + col] = static_cast<float>((row * 16 + col * 4) % 27) * 0.1f;
        }
    }
}

void fill_mask(std::vector<int> &mask, int rows, int cols)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            mask[row * cols + col] = (col == 0 || ((row + col) % 5 != 0)) ? 1 : 0;
        }
    }
}

bool check_output(
    const std::vector<float> &got,
    const std::vector<float> &expected,
    int rows,
    int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        if (std::fabs(got[i] - expected[i]) > 1e-5f)
        {
            std::cerr << "Mismatch" << "\n";
            return false;
        }
    }
    return true;
}

void run_naive_softmax(
    const std::vector<float> &host_input,
    std::vector<float> &host_output,
    int rows,
    int cols)
{
    float *device_input = nullptr;
    float *device_output = nullptr;
    const size_t bytes = host_input.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_naive_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));
    return;
}

void run_online_softmax(
    const std::vector<float> &host_input,
    std::vector<float> &host_output,
    int rows,
    int cols)
{
    float *device_input = nullptr;
    float *device_output = nullptr;
    const size_t bytes = host_input.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_online_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    return;
}

void run_masked_softmax(
    const std::vector<float> &host_input,
    const std::vector<int> &host_mask,
    std::vector<float> &host_output,
    int rows,
    int cols)
{
    float *device_input = nullptr;
    int *device_mask = nullptr;
    float *device_output = nullptr;
    const size_t bytes = host_input.size() * sizeof(float);
    const size_t mask_bytes = host_mask.size() * sizeof(int);

    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_mask, mask_bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_mask, host_mask.data(), mask_bytes, cudaMemcpyHostToDevice));

    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_masked_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_mask, device_output, rows, cols);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_mask));
    CHECK_CUDA(cudaFree(device_output));
}

void run_causal_softmax(
    const std::vector<float> &host_input,
    std::vector<float> &host_output,
    int rows,
    int cols)
{
    float *device_input = nullptr;
    float *device_output = nullptr;
    const size_t bytes = host_input.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_causal_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));
}

void run_block_softmax(
    const std::vector<float> &host_input,
    std::vector<float> &host_output,
    int rows,
    int cols)
{
    float *device_input = nullptr;
    float *device_output = nullptr;
    const size_t bytes = host_input.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = rows;
    softmax_block_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    return;
}

void run_block_online_softmax(
    const std::vector<float> &host_input,
    std::vector<float> &host_output,
    int rows,
    int cols)
{
    float *device_input = nullptr;
    float *device_output = nullptr;
    const size_t bytes = host_input.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = rows;
    softmax_block_online_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    return;
}

void launch_naive_softmax(
    const float *device_input,
    float *device_output,
    int rows,
    int cols)
{
    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_naive_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
}

void launch_online_softmax(
    const float *device_input,
    float *device_output,
    int rows,
    int cols)
{
    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_online_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
}

void launch_masked_softmax(
    const float *device_input,
    const int *device_mask,
    float *device_output,
    int rows,
    int cols)
{
    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_masked_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_mask, device_output, rows, cols);
}

void launch_causal_softmax(
    const float *device_input,
    float *device_output,
    int rows,
    int cols)
{
    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_causal_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
}

void launch_block_softmax(
    const float *device_input,
    float *device_output,
    int rows,
    int cols)
{
    const int blocks = rows;
    softmax_block_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
}

void launch_block_online_softmax(
    const float *device_input,
    float *device_output,
    int rows,
    int cols)
{
    const int blocks = rows;
    softmax_block_online_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
}

template <typename LaunchFunction, typename... Args>
float benchmark_kernel(
    LaunchFunction launch,
    int num_warmups,
    int num_trials,
    Args... args)
{
    double total_ms = 0.0;

    for (int i = 0; i < num_warmups; i++)
    {
        launch(args...);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < num_trials; i++)
    {
        CHECK_CUDA(cudaEventRecord(start));
        launch(args...);
        CHECK_CUDA(cudaEventRecord(stop));
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

int main()
{
    constexpr int rows = 64;
    constexpr int cols = 257;
    constexpr int causal_rows = 64;
    constexpr int causal_cols = 64;

    std::vector<float> host_input(rows * cols);
    std::vector<float> host_output_naive(rows * cols, 0.0f);
    std::vector<float> host_output_online(rows * cols, 0.0f);
    std::vector<int> host_mask(rows * cols);
    std::vector<float> host_output_masked(rows * cols, 0.0f);
    std::vector<float> reference(rows * cols, 0.0f);
    std::vector<float> reference_masked(rows * cols, 0.0f);
    std::vector<float> causal_input(causal_rows * causal_cols);
    std::vector<float> host_output_causal(causal_rows * causal_cols, 0.0f);
    std::vector<float> reference_causal(causal_rows * causal_cols, 0.0f);
    std::vector<float> host_output_block(rows * cols, 0.0f);
    std::vector<float> host_output_block_online(rows * cols, 0.0f);

    // Correctness
    fill_input(host_input, rows, cols);
    softmax_cpu(host_input, reference, rows, cols);
    fill_mask(host_mask, rows, cols);
    softmax_masked_cpu(host_input, host_mask, reference_masked, rows, cols);
    fill_input(causal_input, causal_rows, causal_cols);
    softmax_causal_cpu(causal_input, reference_causal, causal_rows, causal_cols);

    run_naive_softmax(host_input, host_output_naive, rows, cols);
    run_online_softmax(host_input, host_output_online, rows, cols);
    run_masked_softmax(host_input, host_mask, host_output_masked, rows, cols);
    run_causal_softmax(causal_input, host_output_causal, causal_rows, causal_cols);
    run_block_softmax(host_input, host_output_block, rows, cols);
    run_block_online_softmax(host_input, host_output_block_online, rows, cols);

    const bool naive_ok = check_output(host_output_naive, reference, rows, cols);
    const bool online_ok = check_output(host_output_online, reference, rows, cols);
    const bool masked_ok = check_output(host_output_masked, reference_masked, rows, cols);
    const bool causal_ok = check_output(host_output_causal, reference_causal, causal_rows, causal_cols);
    const bool block_ok = check_output(host_output_block, reference, rows, cols);
    const bool block_online_ok = check_output(host_output_block_online, reference, rows, cols);

    if (naive_ok)
    {
        std::cout << "Naive softmax passed!" << "\n";
    }
    else
    {
        std::cout << "Naive softmax failed!" << "\n";
    }

    if (online_ok)
    {
        std::cout << "Online softmax passed!" << "\n";
    }
    else
    {
        std::cout << "Online softmax failed!" << "\n";
    }

    if (masked_ok)
    {
        std::cout << "Masked softmax passed!" << "\n";
    }
    else
    {
        std::cout << "Masked softmax failed!" << "\n";
    }

    if (causal_ok)
    {
        std::cout << "Causal softmax passed!" << "\n";
    }
    else
    {
        std::cout << "Causal softmax failed!" << "\n";
    }
    if (block_ok)
    {
        std::cout << "block softmax passed!" << "\n";
    }
    else
    {
        std::cout << "block softmax failed!" << "\n";
    }
    if (block_online_ok)
    {
        std::cout << "block-online softmax passed!" << "\n";
    }
    else
    {
        std::cout << "block-online softmax failed!" << "\n";
    }

    // benchmarking
    constexpr int num_warmups = 20;
    constexpr int num_trials = 200;
    float *device_input = nullptr;
    float *device_output = nullptr;
    constexpr int benchmark_rows = 16384;
    constexpr int benchmark_cols = 257;

    const size_t input_bytes = static_cast<size_t>(benchmark_rows * benchmark_cols) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&device_output, input_bytes));

    float naive_softmax_ms = benchmark_kernel(launch_naive_softmax, num_warmups, num_trials, device_input, device_output, benchmark_rows, benchmark_cols);
    float online_softmax_ms = benchmark_kernel(launch_online_softmax, num_warmups, num_trials, device_input, device_output, benchmark_rows, benchmark_cols);
    float block_softmax_ms = benchmark_kernel(launch_block_softmax, num_warmups, num_trials, device_input, device_output, benchmark_rows, benchmark_cols);
    float block_online_softmax_ms = benchmark_kernel(launch_block_online_softmax, num_warmups, num_trials, device_input, device_output, benchmark_rows, benchmark_cols);

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    std::cout << "naive softmax: " << naive_softmax_ms << " ms" << "\n";
    std::cout << "online softmax: " << online_softmax_ms << " ms" << "\n";
    std::cout << "block softmax: " << block_softmax_ms << " ms" << "\n";
    std::cout << "block-online softmax: " << block_online_softmax_ms << " ms" << "\n";

    return 0;
}