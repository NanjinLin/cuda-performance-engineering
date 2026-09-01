#include "cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr int kTile = 16;
// 一个block256个thread
// 一个block负责输出C的一个16*16的小块

// 第三版会开始显式引入 warp 思维。
// 我们用一个 32x8 的 block:
// - x 方向 32 个 thread，刚好是一整个 warp 的宽度
// - y 方向 8 个 warp
//
// 这样做的好处是:
// - `threadIdx.y` 可以直接理解成 "第几个 warp"
// - `threadIdx.x` 可以直接理解成 warp 内的 lane id
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kWarpTileN = 64;
constexpr int kBlockTileM = kWarpsPerBlock;
constexpr int kBlockTileN = kWarpTileN;
constexpr int kBlockTileK = 16;

// 第四版再往前走一步:
// 不只是 warp 分工，还让每个 thread 在寄存器里计算一个 2x2 小块。
//
// 这就开始接近现代高性能 GEMM 的核心习惯了:
// - block tile
// - shared-memory tile
// - thread/register tile
//
// 这里 block 仍然是 16x16 = 256 个 thread，
// 但每个 thread 算 2x2 个输出，所以整个 block 会覆盖 32x32 输出 tile。
constexpr int kRegBlockThreadsX = 16;
constexpr int kRegBlockThreadsY = 16;
constexpr int kThreadTileM = 2;
constexpr int kThreadTileN = 2;
constexpr int kRegBlockTileM = kRegBlockThreadsY * kThreadTileM;
constexpr int kRegBlockTileN = kRegBlockThreadsX * kThreadTileN;
constexpr int kRegBlockTileK = 16;

// cpu
void matmul_cpu(
    const std::vector<float> &a,
    const std::vector<float> &b,
    std::vector<float> &c,
    int m, int n, int k)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            float acc = 0.0f;
            for (int inner = 0; inner < k; inner++)
            {
                acc += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

// naive_gpu
// 一个thread负责矩阵中的一个元素 将A的一整行和B的一整列做点积
// 缺点：相邻thread会反复从global memory读取很多重复数据
__global__ void matmul_naive_kernel(
    const float *a,
    const float *b,
    float *c,
    int m, int n, int k)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n)
    {
        return;
    }
    float acc = 0.0f;
    for (int inner = 0; inner < k; inner++)
    {
        acc += a[row * k + inner] * b[inner * n + col];
    }
    c[row * n + col] = acc;
}

// tiled kernel
// 把当前block需要的一小块A和B搬到shared memory，再让block里的thread反复复用
// 依旧是一个thread负责一个位置
__global__ void matmul_tiled_kernel(
    const float *a,
    const float *b,
    float *c,
    int m, int n, int k)
{
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    // 当前thread最终要计算的输出坐标
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;

    // 只要 block 后面存在 __syncthreads()，就不能让 block 中一部分线程提前 return。
    // 第二个原因在于即使它不对应输出，仍可能负责搬运输入数据

    float acc = 0.0f;

    // 每次循环，当前 block 都会:
    // 1. 读入 A 的一个 tile
    // 2. 读入 B 的一个 tile
    // 3. 在 shared memory 里做一小段乘加
    for (int tile_k = 0; tile_k < k; tile_k += kTile)
    {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;
        // 每个元素搬一个A元素和一个B元素进shared memory。越界则补0
        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] =
            (col < n && b_row < k) ? b[b_row * n + col] : 0.0f;

        // 同步:有些thread还在搬数据，其他thread不能读取tile
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kTile; inner++)
        {
            acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < n)
    {
        c[row * n + col] = acc;
    }
}

// warp_kernel
// 变化：
// 1.block内不再是“256个thread平铺干活”，而是显示按warp分工
// 2.每个thread不再只算一个输出，而是在寄存器里累计多个输出
// 分工方式：
// 一个block负责C的一个 8 * 64 tile
// block里8个warp，每个warp负责 1 * 64
// warp里每个lane计算两个值：自己的列和 +32列
__global__ void matmul_warp_tiled_kernel(
    const float *a,
    const float *b,
    float *c,
    int m, int n, int k)
{
    __shared__ float a_tile[kBlockTileM][kBlockTileK];
    __shared__ float b_tile[kBlockTileK][kBlockTileN];

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int row = blockIdx.y * kBlockTileM + warp_id;
    // 每个thread算两个值
    const int col0 = blockIdx.x * kBlockTileN + lane;
    const int col1 = col0 + kWarpSize;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK)
    {
        // 先搬A A tile一共128个元素 启动前128个thread
        if (linear_tid < kBlockTileM * kBlockTileK)
        {
            const int tile_row = linear_tid / kBlockTileK;
            const int tile_col = linear_tid % kBlockTileK;
            const int global_row = blockIdx.y * kBlockTileM + tile_row;
            const int global_col = tile_k + tile_col;

            a_tile[tile_row][tile_col] =
                (global_row < m && global_col < k) ? a[global_row * k + global_col] : 0.0f;
        }
        // 再搬B tile
        // B tile 1024个元素 每个thread搬四个
        for (int idx = linear_tid; idx < kBlockTileK * kBlockTileN; idx += blockDim.x * blockDim.y)
        {
            const int tile_row = idx / kBlockTileN;
            const int tile_col = idx % kBlockTileN;
            const int global_row = tile_k + tile_row;
            const int global_col = blockIdx.x * kBlockTileN + tile_col;
            b_tile[tile_row][tile_col] =
                (global_row < k && global_col < n) ? b[global_row * n + global_col] : 0.0f;
        }

        __syncthreads();
        // 判断一下，不用计算0*
        if (row < m)
        {
#pragma unroll
            for (int inner = 0; inner < kBlockTileK; inner++)
            {
                const float a_value = a_tile[warp_id][inner];
                acc0 += a_value * b_tile[inner][lane];
                acc1 += a_value * b_tile[inner][lane + kWarpSize];
            }
        }
        __syncthreads();
    }
    if (row < m && col0 < n)
    {
        c[row * n + col0] = acc0;
    }
    if (row < m && col1 < n)
    {
        c[row * n + col1] = acc1;
    }
}

// register-blocked matmul kernel
// 每个thread在寄存器维护一个 2*2 的小累加器
// 当一个thread一次算多个输出时，能把A/B数据片在寄存器中复用更多次
// 一个block 16 * 16
// 一个 A_Tile 32 * 16
// 一个 B_Tile 16 * 32
// 更像近现代GEMM骨架：
// block tile -> share-memory tile -> register tile
__global__ void matmul_register_blocked_kernel(
    const float *a,
    const float *b,
    float *c,
    int m, int n, int k)
{
    __shared__ float a_tile[kRegBlockTileM][kRegBlockTileK];
    __shared__ float b_tile[kRegBlockTileK][kRegBlockTileN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row_base = blockIdx.y * kRegBlockTileM + ty * kThreadTileM;
    const int col_base = blockIdx.x * kRegBlockTileN + tx * kThreadTileN;

    float acc[kThreadTileM][kThreadTileN] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int tile_k = 0; tile_k < k; tile_k += kRegBlockTileK)
    {
        // 搬A
        // 每个thread负责搬两行
        const int a_col = tile_k + tx;
        const int a_row0 = blockIdx.y * kRegBlockTileM + ty;
        const int a_row1 = a_row0 + kRegBlockThreadsY;

        a_tile[ty][tx] = (a_row0 < m && a_col < k) ? a[a_row0 * k + a_col] : 0.0f;
        a_tile[ty + kRegBlockThreadsY][tx] = (a_row1 < m && a_col < k) ? a[a_row1 * k + a_col] : 0.0f;

        // 搬B
        // 每个thread负责搬同一行两列
        const int b_row = tile_k + ty;
        const int b_col0 = blockIdx.x * kRegBlockTileN + tx;
        const int b_col1 = b_col0 + kRegBlockThreadsX;

        b_tile[ty][tx] = (b_row < k && b_col0 < n) ? b[b_row * n + b_col0] : 0.0f;
        b_tile[ty][tx + kRegBlockThreadsX] = (b_row < k && b_col1 < n) ? b[b_row * n + b_col1] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kRegBlockTileK; inner++)
        {
            const float a_frag0 = a_tile[ty * kThreadTileM + 0][inner];
            const float a_frag1 = a_tile[ty * kThreadTileM + 1][inner];
            const float b_frag0 = b_tile[inner][tx * kThreadTileN + 0];
            const float b_frag1 = b_tile[inner][tx * kThreadTileN + 1];

            acc[0][0] += a_frag0 * b_frag0;
            acc[0][1] += a_frag0 * b_frag1;
            acc[1][0] += a_frag1 * b_frag0;
            acc[1][1] += a_frag1 * b_frag1;
        }
        __syncthreads();
    }
    // 把 2*2 累加结果写回global memory
    for (int i = 0; i < kThreadTileM; i++)
    {
        for (int j = 0; j < kThreadTileN; j++)
        {
            const int row = row_base + i;
            const int col = col_base + j;
            if (row < m && col < n)
            {
                c[row * n + col] = acc[i][j];
            }
        }
    }
}

void fill_inputs(std::vector<float> &a, std::vector<float> &b, int m, int n, int k)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < k; col++)
        {
            a[row * k + col] = static_cast<float>((row + col) % 41);
        }
    }
    for (int row = 0; row < k; row++)
    {
        for (int col = 0; col < n; col++)
        {
            b[row * n + col] = static_cast<float>((row * 2 + col) % 16);
        }
    }
}

bool check_output(const std::vector<float> &got, const std::vector<float> &expected)
{
    for (size_t i = 0; i < got.size(); i++)
    {
        if (std::fabs(got[i] - expected[i]) > 1e-5f)
        {
            std::cerr << "Mismatch" << "\n";
            return false;
        }
    }
    return true;
}

void run_naive_matmul(
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    std::vector<float> &host_c,
    int m,
    int n,
    int k)
{
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const size_t a_bytes = host_a.size() * sizeof(float);
    const size_t b_bytes = host_b.size() * sizeof(float);
    const size_t c_bytes = host_c.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
    CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
    CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

    const dim3 block(kTile, kTile);
    const dim3 grid(cuda_utils::ceil_div(n, kTile), cuda_utils::ceil_div(m, kTile));
    matmul_naive_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
}

void run_tiled_matmul(
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    std::vector<float> &host_c,
    int m,
    int n,
    int k)
{
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const size_t a_bytes = host_a.size() * sizeof(float);
    const size_t b_bytes = host_b.size() * sizeof(float);
    const size_t c_bytes = host_c.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
    CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
    CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

    const dim3 block(kTile, kTile);
    const dim3 grid(cuda_utils::ceil_div(n, kTile), cuda_utils::ceil_div(m, kTile));
    matmul_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
}

void run_warp_tiled_matmul(
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    std::vector<float> &host_c,
    int m,
    int n,
    int k)
{
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const size_t a_bytes = host_a.size() * sizeof(float);
    const size_t b_bytes = host_b.size() * sizeof(float);
    const size_t c_bytes = host_c.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
    CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
    CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

    // 这一版开始显式按 warp 分工:
    // - 一个 block 是 (32 lanes) x (8 warps)
    // - grid 的一个 block 覆盖输出矩阵里的 8x64 区域
    // 注意这里的两个参数对应threadIdx.x和threadIdx.y 分别是列和行
    const dim3 block(kWarpSize, kWarpsPerBlock);
    const dim3 grid(cuda_utils::ceil_div(n, kBlockTileN), cuda_utils::ceil_div(m, kBlockTileM));
    matmul_warp_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
}

void run_register_blocked_matmul(
    const std::vector<float> &host_a,
    const std::vector<float> &host_b,
    std::vector<float> &host_c,
    int m,
    int n,
    int k)
{
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;

    const size_t a_bytes = host_a.size() * sizeof(float);
    const size_t b_bytes = host_b.size() * sizeof(float);
    const size_t c_bytes = host_c.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
    CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
    CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

    const dim3 block(kRegBlockThreadsX, kRegBlockThreadsY);
    const dim3 grid(
        cuda_utils::ceil_div(n, kRegBlockTileN),
        cuda_utils::ceil_div(m, kRegBlockTileM));
    matmul_register_blocked_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
}

void launch_naive(
    const float *device_a,
    const float *device_b,
    float *device_c,
    int m, int n, int k)
{
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(
        cuda_utils::ceil_div(n, kBlockX),
        cuda_utils::ceil_div(m, kBlockY));
    matmul_naive_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
}

void launch_tiled_kernel(
    const float *device_a,
    const float *device_b,
    float *device_c,
    int m, int n, int k)
{
    const dim3 block(kTile, kTile);
    const dim3 grid(cuda_utils::ceil_div(n, kTile), cuda_utils::ceil_div(m, kTile));
    matmul_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
}

void launch_warp_tiled_kernel(
    const float *device_a,
    const float *device_b,
    float *device_c,
    int m, int n, int k)
{
    const dim3 block(kWarpSize, kWarpsPerBlock);
    const dim3 grid(cuda_utils::ceil_div(n, kBlockTileN), cuda_utils::ceil_div(m, kBlockTileM));
    matmul_warp_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
}
void launch_register_blocked_kernel(
    const float *device_a,
    const float *device_b,
    float *device_c,
    int m, int n, int k)
{
    const dim3 block(kRegBlockThreadsX, kRegBlockThreadsY);
    const dim3 grid(
        cuda_utils::ceil_div(n, kRegBlockTileN),
        cuda_utils::ceil_div(m, kRegBlockTileM));
    matmul_register_blocked_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
}

template <typename LaunchFunction>
float benchmark(
    LaunchFunction launch,
    int num_warmups,
    int num_trials,
    const float *device_a,
    const float *device_b,
    float *device_c,
    int m, int n, int k)
{
    float total_ms = 0.0f;

    for (int i = 0; i < num_warmups; i++)
    {
        launch(device_a, device_b, device_c, m, n, k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < num_trials; i++)
    {
        CHECK_CUDA(cudaEventRecord(start));
        launch(device_a, device_b, device_c, m, n, k);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(
            &elapsed_ms, start, stop));
        total_ms += elapsed_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / num_trials;
}

int main()
{

    constexpr int m = 64;
    constexpr int n = 64;
    constexpr int k = 64;

    std::vector<float> reference(m * n, 0.0f);
    std::vector<float> host_a(m * k);
    std::vector<float> host_b(k * n);
    std::vector<float> host_c_naive(m * n, 0.0f);
    std::vector<float> host_c_tiled(m * n, 0.0f);
    std::vector<float> host_c_warp_tiled(m * n, 0.0f);
    std::vector<float> host_c_register_blocked(m * n, 0.0f);

    fill_inputs(host_a, host_b, m, n, k);
    matmul_cpu(host_a, host_b, reference, m, n, k);

    // correctness
    // naive
    run_naive_matmul(host_a, host_b, host_c_naive, m, n, k);

    // tiled
    run_tiled_matmul(host_a, host_b, host_c_tiled, m, n, k);

    // warp
    run_warp_tiled_matmul(host_a, host_b, host_c_warp_tiled, m, n, k);

    // register
    run_register_blocked_matmul(host_a, host_b, host_c_register_blocked, m, n, k);

    const bool naive_ok = check_output(host_c_naive, reference);
    const bool tiled_ok = check_output(host_c_tiled, reference);
    const bool warp_tiled_ok = check_output(host_c_warp_tiled, reference);
    const bool register_ok = check_output(host_c_register_blocked, reference);

    if (naive_ok)
    {
        std::cout << "naive passed!" << "\n";
    }
    else
    {
        std::cout << "naive failed" << "\n";
    }
    if (tiled_ok)
    {
        std::cout << "tiled passed!" << "\n";
    }
    else
    {
        std::cout << "tiled failed" << "\n";
    }
    if (warp_tiled_ok)
    {
        std::cout << "warp_tield passed!" << "\n";
    }
    else
    {
        std::cout << "warp_tiled failed" << "\n";
    }
    if (register_ok)
    {
        std::cout << "register passed!" << "\n";
    }
    else
    {
        std::cout << "register failed" << "\n";
    }

    // benchmark
    constexpr int num_warmups = 20;
    constexpr int num_trials = 200;
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;
    const int benchmark_m = 1024;
    const int benchmark_n = 1024;
    const int benchmark_k = 1024;

    const size_t a_size = static_cast<size_t>(benchmark_m * benchmark_k) * sizeof(float);
    const size_t b_size = static_cast<size_t>(benchmark_k * benchmark_n) * sizeof(float);
    const size_t c_size = static_cast<size_t>(benchmark_m * benchmark_n) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_a, a_size));
    CHECK_CUDA(cudaMalloc(&device_b, b_size));
    CHECK_CUDA(cudaMalloc(&device_c, c_size));

    std::vector<float> benchmark_a(benchmark_m * benchmark_k);
    std::vector<float> benchmark_b(benchmark_k * benchmark_n);

    fill_inputs(benchmark_a, benchmark_b, benchmark_m, benchmark_n, benchmark_k);

    CHECK_CUDA(cudaMemcpy(device_a, benchmark_a.data(), a_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, benchmark_b.data(), b_size, cudaMemcpyHostToDevice));

    float naive_ms = benchmark(launch_naive, num_warmups, num_trials, device_a, device_b, device_c, benchmark_m, benchmark_n, benchmark_k);
    float tiled_ms = benchmark(launch_tiled_kernel, num_warmups, num_trials, device_a, device_b, device_c, benchmark_m, benchmark_n, benchmark_k);
    float warp_ms = benchmark(launch_warp_tiled_kernel, num_warmups, num_trials, device_a, device_b, device_c, benchmark_m, benchmark_n, benchmark_k);
    float register_ms = benchmark(launch_register_blocked_kernel, num_warmups, num_trials, device_a, device_b, device_c, benchmark_m, benchmark_n, benchmark_k);

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    std::cout << "naive_ms: " << naive_ms << " ms" << "\n";
    std::cout << "tiled_ms: " << tiled_ms << " ms" << "\n";
    std::cout << "warp_ms: " << warp_ms << " ms" << "\n";
    std::cout << "register_ms: " << register_ms << " ms" << "\n";

    return 0;
}