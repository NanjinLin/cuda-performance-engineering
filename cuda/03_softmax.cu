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

// online gpu
__global__ void softmax_online_kernel(
    const float *input,
    float *output,
    int rows,
    int cols)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > rows)
    {
        return;
    }
    const float *row_ptr = input + row * cols;
    float *out_ptr = output + row * cols;

    float row_max = kNegInf;
    double row_sum = 0.0;
    for (int col = 0; col < cols; col++)
    {
        const float x = row_ptr[col];
        const float new_row_max = fmaxf(row_max, x);
        row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) + std::exp(static_cast<double>(x - new_row_max));
        row_max = new_row_max;
    }

    for (int col = 0; col < cols; col++)
    {
        out_ptr[col] = static_cast<float>(std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
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

    const bool naive_ok = check_output(host_output_naive, reference, rows, cols);
    const bool online_ok = check_output(host_output_online, reference, rows, cols);
    const bool masked_ok = check_output(host_output_masked, reference_masked, rows, cols);
    const bool causal_ok = check_output(host_output_causal, reference_causal, causal_rows, causal_cols);

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

    return 0;
}