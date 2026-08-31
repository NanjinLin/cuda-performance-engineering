#include "cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr int kBatch = 4096;
constexpr int kInputDim = 128;
constexpr int kHiddenDim = 256;
constexpr int kOutputDim = 128;
constexpr int kThreadsPerBlock = 256;

constexpr int kTiledThreadsPerBlock = 256;
constexpr int kInputTile = 32;

void fill_inputs(std::vector<float> &x)
{
    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
        x[i] = static_cast<float>((i % 4) * 3 - 2);
    }
}

void fill_weights(std::vector<float> &w1, std::vector<float> &b1,
                  std::vector<float> &w2, std::vector<float> &b2)
{
    for (int i = 0; i < static_cast<int>(w1.size()); ++i)
    {
        w1[i] = static_cast<float>((i % 7) - 3) * 0.1f;
    }
    for (int i = 0; i < static_cast<int>(b1.size()); ++i)
    {
        b1[i] = static_cast<float>((i % 3) - 1) * 0.05f;
    }
    for (int i = 0; i < static_cast<int>(w2.size()); ++i)
    {
        w2[i] = static_cast<float>((i % 5) - 2) * 0.08f;
    }
    for (int i = 0; i < static_cast<int>(b2.size()); ++i)
    {
        b2[i] = static_cast<float>((i % 4) - 1) * 0.03f;
    }
}

__host__ __device__ inline float relu(float value)
{
    return value > 0.0f ? value : 0.0f;
}

void mlp_cpu(
    const std::vector<float> &x,
    const std::vector<float> &w1,
    const std::vector<float> &b1,
    const std::vector<float> &w2,
    const std::vector<float> &b2,
    std::vector<float> &y)
{
    std::vector<float> hidden(kBatch * kHiddenDim, 0.0f);

    for (int batch = 0; batch < kBatch; batch++)
    {
        for (int hidden_idx = 0; hidden_idx < kHiddenDim; hidden_idx++)
        {
            float acc = b1[hidden_idx];
            for (int input_idx = 0; input_idx < kInputDim; input_idx++)
            {
                acc += x[batch * kInputDim + input_idx] * w1[input_idx * kHiddenDim + hidden_idx];
            }
            hidden[batch * kHiddenDim + hidden_idx] = relu(acc);
        }
    }

    for (int batch = 0; batch < kBatch; batch++)
    {
        for (int out_idx = 0; out_idx < kOutputDim; out_idx++)
        {
            float acc = b2[out_idx];
            for (int hidden_idx = 0; hidden_idx < kHiddenDim; hidden_idx++)
            {
                acc += hidden[batch * kHiddenDim + hidden_idx] * w2[hidden_idx * kOutputDim + out_idx];
            }
            y[batch * kOutputDim + out_idx] = acc;
        }
    }
}

// 一个block负责一个batch，一个thread负责一个hidden_idx
__global__ void mlp_naive_linear1_kernel(
    const float *x,
    const float *w1,
    const float *b1,
    float *hidden)
{
    const int batch = blockIdx.x;
    const int hidden_idx = threadIdx.x;
    if (batch >= kBatch || hidden_idx >= kHiddenDim)
    {
        return;
    }
    float acc = b1[hidden_idx];
    for (int input_idx = 0; input_idx < kInputDim; input_idx++)
    {
        acc += x[batch * kInputDim + input_idx] * w1[input_idx * kHiddenDim + hidden_idx];
    }
    hidden[batch * kHiddenDim + hidden_idx] = acc;
}

__global__ void relu_kernel(float *values, int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        values[index] = relu(values[index]);
    }
}

__global__ void mlp_naive_linear2_kernel(
    const float *hidden,
    const float *w2,
    const float *b2,
    float *y)
{
    const int batch = blockIdx.x;
    const int out_idx = threadIdx.x;
    if (batch >= kBatch || out_idx >= kOutputDim)
    {
        return;
    }
    float acc = b2[out_idx];
    for (int hidden_idx = 0; hidden_idx < kHiddenDim; hidden_idx++)
    {
        acc += hidden[batch * kHiddenDim + hidden_idx] * w2[hidden_idx * kOutputDim + out_idx];
    }
    y[batch * kOutputDim + out_idx] = acc;
}

// fused MLP
// 把第一层linear和relu融合在一起
// 核心：在写出中间值之前，直接把激活做了
// 第二层不变
__global__ void mlp_fused_linear1_relu_kernel(
    const float *x,
    const float *w1,
    const float *b1,
    float *hidden)
{
    const int batch = blockIdx.x;
    const int hidden_idx = threadIdx.x;
    if (batch >= kBatch || hidden_idx >= kHiddenDim)
    {
        return;
    }

    float acc = b1[hidden_idx];
    for (int input_idx = 0; input_idx < kInputDim; input_idx++)
    {
        acc += x[batch * kInputDim + input_idx] * w1[input_idx * kHiddenDim + hidden_idx];
    }
    hidden[batch * kHiddenDim + hidden_idx] = relu(acc);
}

// tiled MLP
// 一个Block负责一个batch
// 把输入向量 x 分 tile 搬进 shared memory，
// 隐藏层在同一个 kernel 里完成线性变换 + ReLU
// 隐藏层结果留在 shared memory 中，直接做第二层输出
// 真正的GPU kernel思维：少一次全局内存往返，输入数据可以被多个 hidden thread 复用
// 输出 epilogue 直接在 kernel 里写回，不需要分成两个linear_kernel
__global__ void mlp_tiled_fused_kernel(
    const float *x,
    const float *w1,
    const float *b1,
    const float *w2,
    const float *b2,
    float *y)
{
    __shared__ float x_tile[kInputDim];
    __shared__ float hidden_shared[kHiddenDim];

    const int batch = blockIdx.x;
    const int tid = threadIdx.x;
    if (batch >= kBatch)
    {
        return;
    }

    float hidden_acc = 0.0f;
    if (tid < kHiddenDim)
    {
        hidden_acc = b1[tid];
    }

    for (int tile = 0; tile < kInputDim; tile += kInputTile)
    {
        if (tid < kInputTile)
        {
            const int input_idx = tile + tid;
            x_tile[tid] = (input_idx < kInputDim) ? x[batch * kInputDim + input_idx] : 0.0f;
        }
        __syncthreads();
        if (tid < kHiddenDim)
        {
#pragma unroll
            for (int i = 0; i < kInputTile; i++)
            {
                const int input_idx = tile + i;
                if (input_idx < kInputDim)
                {
                    hidden_acc += x_tile[i] * w1[input_idx * kHiddenDim + tid];
                }
            }
        }
        __syncthreads();
    }

    if (tid < kHiddenDim)
    {
        hidden_shared[tid] = relu(hidden_acc);
    }
    __syncthreads();

    if (tid < kOutputDim)
    {
        float acc = b2[tid];
        for (int hidden_idx = 0; hidden_idx < kHiddenDim; hidden_idx++)
        {
            acc += hidden_shared[hidden_idx] * w2[hidden_idx * kOutputDim + tid];
        }
        y[batch * kOutputDim + tid] = acc;
    }
}

void run_naive_mlp(
    const std::vector<float> &x,
    const std::vector<float> &w1,
    const std::vector<float> &b1,
    const std::vector<float> &w2,
    const std::vector<float> &b2,
    std::vector<float> &y)
{
    float *device_x = nullptr;
    float *device_w1 = nullptr;
    float *device_b1 = nullptr;
    float *device_w2 = nullptr;
    float *device_b2 = nullptr;
    float *device_hidden = nullptr;
    float *device_y = nullptr;

    CHECK_CUDA(cudaMalloc(&device_x, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_w1, w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b1, b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_w2, w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b2, b2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_hidden, kBatch * kHiddenDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_y, y.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(device_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    mlp_naive_linear1_kernel<<<kBatch, kHiddenDim>>>(device_x, device_w1, device_b1, device_hidden);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    relu_kernel<<<cuda_utils::ceil_div(kBatch * kHiddenDim, kThreadsPerBlock), kThreadsPerBlock>>>(
        device_hidden, kBatch * kHiddenDim);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    mlp_naive_linear2_kernel<<<kBatch, kOutputDim>>>(device_hidden, device_w2, device_b2, device_y);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(y.data(), device_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(device_w1));
    CHECK_CUDA(cudaFree(device_b1));
    CHECK_CUDA(cudaFree(device_w2));
    CHECK_CUDA(cudaFree(device_b2));
    CHECK_CUDA(cudaFree(device_hidden));
    CHECK_CUDA(cudaFree(device_y));
}

void run_fused_mlp(
    const std::vector<float> &x,
    const std::vector<float> &w1,
    const std::vector<float> &b1,
    const std::vector<float> &w2,
    const std::vector<float> &b2,
    std::vector<float> &y)
{
    float *device_x = nullptr;
    float *device_w1 = nullptr;
    float *device_b1 = nullptr;
    float *device_w2 = nullptr;
    float *device_b2 = nullptr;
    float *device_hidden = nullptr;
    float *device_y = nullptr;

    CHECK_CUDA(cudaMalloc(&device_x, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_w1, w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b1, b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_w2, w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b2, b2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_hidden, kBatch * kHiddenDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_y, y.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(device_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Fused 版本把第一层的线性变换和 ReLU 放在同一个 kernel 里。
    // 这是一种很常见的优化思路：减少中间结果的读写次数。
    mlp_fused_linear1_relu_kernel<<<kBatch, kHiddenDim>>>(device_x, device_w1, device_b1, device_hidden);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    mlp_naive_linear2_kernel<<<kBatch, kOutputDim>>>(device_hidden, device_w2, device_b2, device_y);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(y.data(), device_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(device_w1));
    CHECK_CUDA(cudaFree(device_b1));
    CHECK_CUDA(cudaFree(device_w2));
    CHECK_CUDA(cudaFree(device_b2));
    CHECK_CUDA(cudaFree(device_hidden));
    CHECK_CUDA(cudaFree(device_y));
}

void run_tiled_fused_mlp(
    const std::vector<float> &x,
    const std::vector<float> &w1,
    const std::vector<float> &b1,
    const std::vector<float> &w2,
    const std::vector<float> &b2,
    std::vector<float> &y)
{
    float *device_x = nullptr;
    float *device_w1 = nullptr;
    float *device_b1 = nullptr;
    float *device_w2 = nullptr;
    float *device_b2 = nullptr;
    float *device_y = nullptr;

    CHECK_CUDA(cudaMalloc(&device_x, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_w1, w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b1, b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_w2, w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b2, b2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_y, y.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(device_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    mlp_tiled_fused_kernel<<<kBatch, kTiledThreadsPerBlock>>>(
        device_x, device_w1, device_b1, device_w2, device_b2, device_y);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(y.data(), device_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(device_w1));
    CHECK_CUDA(cudaFree(device_b1));
    CHECK_CUDA(cudaFree(device_w2));
    CHECK_CUDA(cudaFree(device_b2));
    CHECK_CUDA(cudaFree(device_y));
}

bool check_output(
    const std::vector<float> &got,
    const std::vector<float> &expected)
{
    for (size_t i = 0; i < got.size(); i++)
    {
        if (std::fabs(got[i] - expected[i]) > 1e-5)
        {
            std::cerr << "Mismatched X" << "\n";
            return false;
        }
    }
    return true;
}

void launch_naive_kernel(
    const float *x,
    const float *w1,
    const float *b1,
    const float *w2,
    const float *b2,
    float *hidden,
    float *y)
{
    const int blocks = kBatch;
    const int hidden_count = kBatch * kHiddenDim;
    const int relu_blocks =
        cuda_utils::ceil_div(hidden_count, kThreadsPerBlock);
    mlp_naive_linear1_kernel<<<blocks, kHiddenDim>>>(x, w1, b1, hidden);
    relu_kernel<<<relu_blocks, kThreadsPerBlock>>>(
        hidden,
        hidden_count);
    mlp_naive_linear2_kernel<<<blocks, kOutputDim>>>(hidden, w2, b2, y);
}

void launch_fused_kernel(
    const float *x,
    const float *w1,
    const float *b1,
    const float *w2,
    const float *b2,
    float *hidden,
    float *y)
{
    const int blocks = kBatch;
    mlp_fused_linear1_relu_kernel<<<blocks, kThreadsPerBlock>>>(x, w1, b1, hidden);
    mlp_naive_linear2_kernel<<<blocks, kOutputDim>>>(hidden, w2, b2, y);
}

void launch_tiled_kernel(
    const float *x,
    const float *w1,
    const float *b1,
    const float *w2,
    const float *b2,
    float *y)
{
    const int blocks = kBatch;
    mlp_tiled_fused_kernel<<<blocks, kTiledThreadsPerBlock>>>(x, w1, b1, w2, b2, y);
}

template <typename LaunchFunction, typename... Args>
float benchmark(
    LaunchFunction launch,
    const int num_warmups,
    const int num_trials,
    Args... args)
{
    float total_ms = 0.0f;

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
    std::vector<float> x(kBatch * kInputDim);
    std::vector<float> w1(kInputDim * kHiddenDim);
    std::vector<float> b1(kHiddenDim);
    std::vector<float> w2(kHiddenDim * kOutputDim);
    std::vector<float> b2(kOutputDim);

    std::vector<float> naive_y(kBatch * kOutputDim, 0.0f);
    std::vector<float> fused_y(kBatch * kOutputDim, 0.0f);
    std::vector<float> tiled_fused_y(kBatch * kOutputDim, 0.0f);
    std::vector<float> reference_y(kBatch * kOutputDim, 0.0f);

    fill_inputs(x);
    fill_weights(w1, b1, w2, b2);

    mlp_cpu(x, w1, b1, w2, b2, reference_y);
    run_naive_mlp(x, w1, b1, w2, b2, naive_y);
    run_fused_mlp(x, w1, b1, w2, b2, fused_y);
    run_tiled_fused_mlp(x, w1, b1, w2, b2, tiled_fused_y);

    const bool naive_ok = check_output(naive_y, reference_y);
    const bool fused_ok = check_output(fused_y, reference_y);
    const bool tiled_fused_ok = check_output(tiled_fused_y, reference_y);

    if (naive_ok)
    {
        std::cout << "Naive MLP passed!" << "\n";
    }
    else
    {
        std::cout << "Naive MLP failed!" << "\n";
    }
    if (fused_ok)
    {
        std::cout << "Fused MLP passed!" << "\n";
    }
    else
    {
        std::cout << "Fused MLP failed!" << "\n";
    }
    if (tiled_fused_ok)
    {
        std::cout << "Tiled_fused MLP passed!" << "\n";
    }
    else
    {
        std::cout << "Tiled_fused MLP failed!" << "\n";
    }

    // benchmark
    constexpr int num_warmups = 20;
    constexpr int num_trials = 200;
    /*
    kBatch = 4096;
    kInputDim = 128;
    kHiddenDim = 256;
    kOutputDim = 128;
    kThreadsPerBlock = 256;
    kTiledThreadsPerBlock = 256;
    kInputTile = 32;
    */

    float *device_x = nullptr;
    float *device_w1 = nullptr;
    float *device_b1 = nullptr;
    float *device_w2 = nullptr;
    float *device_b2 = nullptr;
    float *device_hidden = nullptr;
    float *device_y = nullptr;

    const size_t x_size = static_cast<size_t>(kBatch * kInputDim) * sizeof(float);
    const size_t w1_size = static_cast<size_t>(kInputDim * kHiddenDim) * sizeof(float);
    const size_t b1_size = static_cast<size_t>(kHiddenDim) * sizeof(float);
    const size_t w2_size = static_cast<size_t>(kHiddenDim * kOutputDim) * sizeof(float);
    const size_t b2_size = static_cast<size_t>(kOutputDim) * sizeof(float);
    const size_t hidden_size = static_cast<size_t>(kBatch * kHiddenDim) * sizeof(float);
    const size_t y_size = static_cast<size_t>(kBatch * kOutputDim) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_x, x_size));
    CHECK_CUDA(cudaMalloc(&device_w1, w1_size));
    CHECK_CUDA(cudaMalloc(&device_b1, b1_size));
    CHECK_CUDA(cudaMalloc(&device_w2, w2_size));
    CHECK_CUDA(cudaMalloc(&device_b2, b2_size));
    CHECK_CUDA(cudaMalloc(&device_hidden, hidden_size));
    CHECK_CUDA(cudaMalloc(&device_y, y_size));

    CHECK_CUDA(cudaMemcpy(device_x, x.data(), x_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_w1, w1.data(), w1_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b1, b1.data(), b1_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_w2, w2.data(), w2_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b2, b2.data(), b2_size, cudaMemcpyHostToDevice));

    float naive_ms = benchmark(launch_naive_kernel, num_warmups, num_trials, device_x, device_w1, device_b1, device_w2, device_b2, device_hidden, device_y);
    float fused_ms = benchmark(launch_fused_kernel, num_warmups, num_trials, device_x, device_w1, device_b1, device_w2, device_b2, device_hidden, device_y);
    float tiled_ms = benchmark(launch_tiled_kernel, num_warmups, num_trials, device_x, device_w1, device_b1, device_w2, device_b2, device_y);

    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(device_w1));
    CHECK_CUDA(cudaFree(device_b1));
    CHECK_CUDA(cudaFree(device_w2));
    CHECK_CUDA(cudaFree(device_b2));
    CHECK_CUDA(cudaFree(device_hidden));
    CHECK_CUDA(cudaFree(device_y));

    std::cout << "naive_ms: " << naive_ms << " ms" << "\n";
    std::cout << "fused_ms: " << fused_ms << " ms" << "\n";
    std::cout << "tiled_ms: " << tiled_ms << " ms" << "\n";

    return 0;
}