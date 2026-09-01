#include "cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr int kSeqLen = 512;
constexpr int kHeadDim = 64;
constexpr int kTileTokens = 32;
constexpr float kScale = 1.0f / 8.0f;

constexpr int kThreadsPerBlock = 256;

double dot_row(
    const std::vector<float> &a,
    int a_row,
    const std::vector<float> &b,
    int b_row)
{
    double acc = 0.0;
    for (int d = 0; d < kHeadDim; d++)
    {
        acc += static_cast<double>(a[a_row * kHeadDim + d]) * static_cast<double>(b[b_row * kHeadDim + d]);
    }
    return acc;
}

void attention_cpu(
    const std::vector<float> &q,
    const std::vector<float> &k,
    const std::vector<float> &v,
    std::vector<float> &out,
    bool causal)
{
    std::vector<double> scores(kSeqLen * kSeqLen, 0.0);
    std::vector<double> probs(kSeqLen * kSeqLen, 0.0);
    for (int row = 0; row < kSeqLen; row++)
    {
        double row_max = -1e30;
        for (int col = 0; col < kSeqLen; col++)
        {
            const bool allowed = !causal || col <= row;
            const double score = allowed ? dot_row(q, row, k, col) * static_cast<double>(kScale) : -1e30;
            scores[row * kSeqLen + col] = score;
            row_max = std::max(row_max, score);
        }
        double row_sum = 0.0;
        for (int col = 0; col < kSeqLen; col++)
        {
            const double value = std::exp(scores[row * kSeqLen + col] - row_max);
            probs[row * kSeqLen + col] = value;
            row_sum += value;
        }
        for (int d = 0; d < kHeadDim; d++)
        {
            double acc = 0.0;
            for (int col = 0; col < kSeqLen; col++)
            {
                const double p = probs[row * kSeqLen + col] / row_sum;
                acc += p * static_cast<double>(v[col * kHeadDim + d]);
            }
            out[row * kHeadDim + d] = static_cast<float>(acc);
        }
    }
}

// naive
__global__ void naive_attention_scores_kernel(
    const float *q,
    const float *k,
    float *scores,
    bool causal)
{
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    if (row >= kSeqLen || col >= kSeqLen)
        return;
    if (!causal || col <= row)
    {
        float acc = 0.0f;
        for (int d = 0; d < kHeadDim; d++)
        {
            acc += q[row * kHeadDim + d] * k[col * kHeadDim + d];
        }
        scores[row * kSeqLen + col] = acc * kScale;
    }
    else
    {
        scores[row * kSeqLen + col] = -1e30f;
    }
}

/*
__global__ void naive_attention_softmax_kernel(float *scores)
{
    const int row = blockIdx.x;
    if (row >= kSeqLen)
    {
        return;
    }

    __shared__ float shared[kSeqLen];
    const int col = threadIdx.x;

    shared[col] = scores[row * kSeqLen + col];
    __syncthreads();

    // 先找这一行的最大值，避免 exp 之后数值爆掉。
    float row_max = shared[0];
    for (int i = 1; i < kSeqLen; ++i)
    {
        row_max = fmaxf(row_max, shared[i]);
    }

    float row_sum = 0.0f;
    for (int i = 0; i < kSeqLen; ++i)
    {
        shared[i] = expf(shared[i] - row_max);
        row_sum += shared[i];
    }

    scores[row * kSeqLen + col] = shared[col] / row_sum;
}
    */

__global__ void naive_attention_softmax_kernel(
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

    float local_max = -1e30f;
    float local_sum = 0.0f;

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

    for (int col = tid; col < cols; col += blockDim.x)
    {
        out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
    }
}

__global__ void naive_attention_value_kernel(
    const float *scores,
    const float *v,
    float *out)
{
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    if (row >= kSeqLen || d >= kHeadDim)
        return;

    float acc = 0.0f;
    for (int col = 0; col < kSeqLen; ++col)
    {
        acc += scores[row * kSeqLen + col] * v[col * kHeadDim + d];
    }
    out[row * kHeadDim + d] = acc;
}

// tiled attention
// 不把scores完整写出来
__global__ void attention_tiled_kernel(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    bool causal)
{
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    if (row >= kSeqLen || d >= kHeadDim)
        return;

    __shared__ float q_shared[kHeadDim];
    __shared__ float k_tile[kTileTokens][kHeadDim];
    __shared__ float v_tile[kTileTokens][kHeadDim];
    __shared__ float scores[kTileTokens];
    __shared__ float weights[kTileTokens];
    __shared__ float running_max_shared;
    __shared__ float running_sum_shared;
    __shared__ float tile_max_shared;
    __shared__ float tile_sum_shared;
    __shared__ float old_scale_shared;

    if (d < kHeadDim)
    {
        q_shared[d] = q[row * kHeadDim + d];
    }

    __syncthreads();

    // online softmax
    if (d == 0)
    {
        running_max_shared = -1e30f;
        running_sum_shared = 0.0f;
    }

    __syncthreads();

    float acc = 0.0f;
    // 块内搬数据 -> 算score —> 更新softmax
    for (int tile_start = 0; tile_start < kSeqLen; tile_start += kTileTokens)
    {
        // 搬数据
        for (int token = 0; token < kTileTokens; token++)
        {
            const int seq_idx = tile_start + token;
            if (seq_idx < kSeqLen)
            {
                k_tile[token][d] = k[seq_idx * kHeadDim + d];
                v_tile[token][d] = v[seq_idx * kHeadDim + d];
            }
            else
            {
                k_tile[token][d] = 0.0f;
                v_tile[token][d] = 0.0f;
            }
        }
        __syncthreads();

        // 只有一个thread做运算，算出这一块的score、max、sum，再广播给其他thread
        if (d == 0)
        {
            // 计算 + max
            tile_max_shared = -1e30f;
            for (int token = 0; token < kTileTokens; token++)
            {
                const int seq_idx = tile_start + token;
                const bool allowed = seq_idx < kSeqLen && (!causal || seq_idx <= row);
                if (allowed)
                {
                    float score = 0.0f;
                    for (int i = 0; i < kHeadDim; i++)
                    {
                        score += q_shared[i] * k_tile[token][i];
                    }
                    score *= kScale;
                    scores[token] = score;
                    tile_max_shared = fmaxf(tile_max_shared, score);
                }
                else
                {
                    scores[token] = -1e30f;
                }
            }
            // sum
            tile_sum_shared = 0.0f;
            for (int token = 0; token < kTileTokens; token++)
            {
                if (tile_start + token < kSeqLen)
                {
                    tile_sum_shared += expf(scores[token] - tile_max_shared);
                }
            }
            // 更新
            const float new_max = fmaxf(running_max_shared, tile_max_shared);
            old_scale_shared = expf(running_max_shared - new_max);
            running_sum_shared = running_sum_shared * old_scale_shared + tile_sum_shared * expf(tile_max_shared - new_max);
            running_max_shared = new_max;

            for (int token = 0; token < kTileTokens; token++)
            {
                if (tile_start + token < kSeqLen)
                {
                    weights[token] = expf(scores[token] - new_max);
                }
                else
                {
                    weights[token] = 0.0f;
                }
            }
        }
        __syncthreads();

        acc = acc * old_scale_shared;
        for (int token = 0; token < kTileTokens; token++)
        {
            acc += weights[token] * v_tile[token][d];
        }
        __syncthreads();
    }
    out[row * kHeadDim + d] = acc / running_sum_shared;
}

void fill_inputs(std::vector<float> &q, std::vector<float> &k, std::vector<float> &v)
{
    for (int i = 0; i < kSeqLen * kHeadDim; ++i)
    {
        q[i] = static_cast<float>((i % 5) - 2) * 0.2f;
        k[i] = static_cast<float>((i % 7) - 3) * 0.15f;
        v[i] = static_cast<float>((i % 6) - 2) * 0.1f;
    }
}

bool check_output(
    const std::vector<float> &got, const std::vector<float> &expected)
{
    for (size_t i = 0; i < got.size(); i++)
    {
        if (std::fabs(got[i] - expected[i]) > 1e-5f)
        {
            std::cerr << "Mismatched!" << "\n";
            return false;
        }
    }
    return true;
}

void run_naive_attention(
    const std::vector<float> &q,
    const std::vector<float> &k,
    const std::vector<float> &v,
    std::vector<float> &out,
    bool causal)
{
    float *device_q = nullptr;
    float *device_k = nullptr;
    float *device_v = nullptr;
    float *device_scores = nullptr;
    float *device_out = nullptr;

    CHECK_CUDA(cudaMalloc(&device_q, q.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_k, k.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_v, v.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_scores, kSeqLen * kSeqLen * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_out, out.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(device_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

    naive_attention_scores_kernel<<<kSeqLen, kSeqLen>>>(device_q, device_k, device_scores, causal);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    naive_attention_softmax_kernel<<<kSeqLen, kThreadsPerBlock>>>(device_scores, device_scores, kSeqLen, kSeqLen);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    naive_attention_value_kernel<<<kSeqLen, kHeadDim>>>(device_scores, device_v, device_out);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out.data(), device_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_q));
    CHECK_CUDA(cudaFree(device_k));
    CHECK_CUDA(cudaFree(device_v));
    CHECK_CUDA(cudaFree(device_scores));
    CHECK_CUDA(cudaFree(device_out));
}

void run_tiled_attention(
    const std::vector<float> &q,
    const std::vector<float> &k,
    const std::vector<float> &v,
    std::vector<float> &out,
    bool causal)
{
    float *device_q = nullptr;
    float *device_k = nullptr;
    float *device_v = nullptr;
    float *device_out = nullptr;

    CHECK_CUDA(cudaMalloc(&device_q, q.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_k, k.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_v, v.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_out, out.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(device_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

    attention_tiled_kernel<<<kSeqLen, kHeadDim>>>(device_q, device_k, device_v, device_out, causal);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out.data(), device_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_q));
    CHECK_CUDA(cudaFree(device_k));
    CHECK_CUDA(cudaFree(device_v));
    CHECK_CUDA(cudaFree(device_out));
}

void launch_naive_kernel(
    const float *q,
    const float *k,
    const float *v,
    float *scores,
    float *out)
{
    naive_attention_scores_kernel<<<kSeqLen, kSeqLen>>>(q, k, scores, false);
    naive_attention_softmax_kernel<<<kSeqLen, kThreadsPerBlock>>>(scores, scores, kSeqLen, kSeqLen);
    naive_attention_value_kernel<<<kSeqLen, kHeadDim>>>(scores, v, out);
}

void launch_tiled_kernel(
    const float *q,
    const float *k,
    const float *v,
    float *scores,
    float *out)
{
    attention_tiled_kernel<<<kSeqLen, kHeadDim>>>(q, k, v, out, false);
}

template <typename LaunchFunction>
float benchmark(
    LaunchFunction launch,
    const int num_warmups,
    const int num_trials,
    const float *q,
    const float *k,
    const float *v,
    float *out,
    float *scores)
{
    float total_ms = 0.0f;

    for (int i = 0; i < num_warmups; i++)
    {
        launch(q, k, v, scores, out);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < num_trials; i++)
    {
        CHECK_CUDA(cudaEventRecord(start));
        launch(q, k, v, scores, out);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / num_trials;
}

int main()
{
    std::vector<float> q(kSeqLen * kHeadDim);
    std::vector<float> k(kSeqLen * kHeadDim);
    std::vector<float> v(kSeqLen * kHeadDim);
    std::vector<float> naive_out(kSeqLen * kHeadDim, 0.0f);
    std::vector<float> tiled_out(kSeqLen * kHeadDim, 0.0f);
    std::vector<float> causal_naive_out(kSeqLen * kHeadDim, 0.0f);
    std::vector<float> causal_tiled_out(kSeqLen * kHeadDim, 0.0f);
    std::vector<float> reference_out(kSeqLen * kHeadDim, 0.0f);
    std::vector<float> causal_reference_out(kSeqLen * kHeadDim, 0.0f);

    fill_inputs(q, k, v);
    attention_cpu(q, k, v, reference_out, false);
    attention_cpu(q, k, v, causal_reference_out, true);
    run_naive_attention(q, k, v, naive_out, false);
    run_tiled_attention(q, k, v, tiled_out, false);
    run_naive_attention(q, k, v, causal_naive_out, true);
    run_tiled_attention(q, k, v, causal_tiled_out, true);

    // Correctness
    const bool naive_ok = check_output(naive_out, reference_out);
    const bool tiled_ok = check_output(tiled_out, reference_out);
    const bool causal_naive_ok = check_output(causal_naive_out, causal_reference_out);
    const bool causal_tiled_ok = check_output(causal_tiled_out, causal_reference_out);

    if (naive_ok)
    {
        std::cout << "Naive passed!" << "\n";
    }
    else
    {
        std::cout << "Naive failed!" << "\n";
    }
    if (tiled_ok)
    {
        std::cout << "Tiled passed!" << "\n";
    }
    else
    {
        std::cout << "Tiled failed!" << "\n";
    }
    if (causal_naive_ok)
    {
        std::cout << "Causal Naive passed!" << "\n";
    }
    else
    {
        std::cout << "Causal Naive failed!" << "\n";
    }
    if (causal_tiled_ok)
    {
        std::cout << "Causal Tiled passed!" << "\n";
    }
    else
    {
        std::cout << "Causal Tiled failed!" << "\n";
    }

    // benchmark
    const int num_warmups = 20;
    const int num_trials = 200;

    float *device_q = nullptr;
    float *device_k = nullptr;
    float *device_v = nullptr;
    float *device_out = nullptr;
    float *scores = nullptr;

    CHECK_CUDA(cudaMalloc(&device_q, q.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_k, k.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_v, v.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_out, reference_out.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&scores, kSeqLen * kSeqLen * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(device_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

    float naive_ms = benchmark(
        launch_naive_kernel,
        num_warmups,
        num_trials,
        device_q,
        device_k,
        device_v,
        device_out,
        scores);

    float tiled_ms = benchmark(
        launch_tiled_kernel,
        num_warmups,
        num_trials,
        device_q,
        device_k,
        device_v,
        device_out,
        scores);

    std::cout << "Naive_ms: " << naive_ms << " ms" << "\n";
    std::cout << "Tiled_ms: " << tiled_ms << " ms" << "\n";

    CHECK_CUDA(cudaFree(device_q));
    CHECK_CUDA(cudaFree(device_k));
    CHECK_CUDA(cudaFree(device_v));
    CHECK_CUDA(cudaFree(device_out));
    CHECK_CUDA(cudaFree(scores));

    return 0;
}