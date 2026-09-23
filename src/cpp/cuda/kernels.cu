#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include "../utils/types.hpp"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err__));             \
        }                                                                    \
    } while (0)

DeviceWorkspace::DeviceWorkspace() = default;

DeviceWorkspace::~DeviceWorkspace() {
    release();
}

void DeviceWorkspace::allocate(std::size_t n) {
    const int block = 256;
    grid_size = static_cast<int>(
        std::min<std::size_t>((n + block - 1) / block, 4096));

    CUDA_CHECK(cudaMalloc((void**)&d_blockMinInt,   grid_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockMaxInt,   grid_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_minInt,        sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxInt,        sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&d_blockMin,      grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockMax,      grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockSum,      grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockSumSq,    grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockChecksum, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_blockSqDiff,   grid_size * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&d_min,      sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_max,      sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_sum,      sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_sumSq,    sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_checksum, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_sqDiff,   sizeof(double)));
}

void DeviceWorkspace::release() {
    auto safeFree = [](auto*& p) {
        if (p) { cudaFree(p); p = nullptr; }
    };
    safeFree(d_blockMinInt);
    safeFree(d_blockMaxInt);
    safeFree(d_minInt);
    safeFree(d_maxInt);
    safeFree(d_blockMin);
    safeFree(d_blockMax);
    safeFree(d_blockSum);
    safeFree(d_blockSumSq);
    safeFree(d_blockChecksum);
    safeFree(d_blockSqDiff);
    safeFree(d_min);
    safeFree(d_max);
    safeFree(d_sum);
    safeFree(d_sumSq);
    safeFree(d_checksum);
    safeFree(d_sqDiff);
}

// -------------------------------------------------------------------------
// FractalWeights computation
// -------------------------------------------------------------------------
__global__ void computeFractalWeightsKernel(
    int* __restrict__ weightField,
    std::size_t totalCells, std::size_t width, std::size_t height,
    double x0, double y0, double dx, double dy, int maxFractalIterations)
{
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalCells; idx += gridDim.x * blockDim.x)
    {
        const std::size_t i = idx % width;
        const std::size_t j = idx / width;

        const double cReal = x0 + dx * static_cast<double>(i);
        const double cImag = y0 + dy * static_cast<double>(j);

        double zReal = 0.0;
        double zImag = 0.0;
        int iter = 0;

        for (; iter < maxFractalIterations; ++iter) {
            if (zReal * zReal + zImag * zImag > 4.0) break;
            const double tmp = zReal * zReal - zImag * zImag + cReal;
            zImag = 2.0 * zReal * zImag + cImag;
            zReal = tmp;
        }

        weightField[idx] = iter;
    }
}

void computeFractalWeightsCUDA(
    int* d_weightField,
    const SimulationConfig& cfg,
    const GridMapping& mapping)
{
    const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);
    const int block = 256;
    const int grid = static_cast<int>(
        std::min<std::size_t>((totalCells + block - 1) / block, 4096));

    computeFractalWeightsKernel<<<grid, block>>>(
        d_weightField, totalCells, cfg.gridWidth, cfg.gridHeight,
        mapping.x0, mapping.y0, mapping.dx, mapping.dy, cfg.maxFractalIterations);

    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------------------------
// Compute weights range
// -------------------------------------------------------------------------
__global__ void minMaxReduceKernel(
    const int* __restrict__ data, std::size_t n,
    int* __restrict__ blockMin, int* __restrict__ blockMax)
{
    extern __shared__ int smem[];
    int* sMin = smem;
    int* sMax = smem + blockDim.x;

    const int t = threadIdx.x;
    int localMin = INT_MAX;
    int localMax = INT_MIN;

    for (std::size_t i = blockIdx.x * blockDim.x + t;
         i < n; i += gridDim.x * blockDim.x)
    {
        const int v = data[i];
        localMin = min(localMin, v);
        localMax = max(localMax, v);
    }

    sMin[t] = localMin;
    sMax[t] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sMin[t] = min(sMin[t], sMin[t + stride]);
            sMax[t] = max(sMax[t], sMax[t + stride]);
        }
        __syncthreads();
    }

    if (t == 0) {
        blockMin[blockIdx.x] = sMin[0];
        blockMax[blockIdx.x] = sMax[0];
    }
}

__global__ void minMaxFinalKernel(
    const int* __restrict__ blockMin, const int* __restrict__ blockMax,
    int numBlocks, int* __restrict__ outMin, int* __restrict__ outMax)
{
    extern __shared__ int smem[];
    int* sMin = smem;
    int* sMax = smem + blockDim.x;

    const int t = threadIdx.x;
    int localMin = INT_MAX;
    int localMax = INT_MIN;

    for (int i = t; i < numBlocks; i += blockDim.x) {
        localMin = min(localMin, blockMin[i]);
        localMax = max(localMax, blockMax[i]);
    }

    sMin[t] = localMin;
    sMax[t] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sMin[t] = min(sMin[t], sMin[t + stride]);
            sMax[t] = max(sMax[t], sMax[t + stride]);
        }
        __syncthreads();
    }

    if (t == 0) {
        *outMin = sMin[0];
        *outMax = sMax[0];
    }
}

std::pair<int, int> computeWeightRangeCUDA(
    const int* d_weightField,
    std::size_t totalCells,
    const DeviceWorkspace& ws)
{
    const int block = 256;

    minMaxReduceKernel<<<ws.grid_size, block, 2 * block * sizeof(int)>>>(
        d_weightField, totalCells, ws.d_blockMinInt, ws.d_blockMaxInt);
    CUDA_CHECK(cudaGetLastError());

    minMaxFinalKernel<<<1, block, 2 * block * sizeof(int)>>>(
        ws.d_blockMinInt, ws.d_blockMaxInt, ws.grid_size, ws.d_minInt, ws.d_maxInt);
    CUDA_CHECK(cudaGetLastError());

    int minVal = 0;
    int maxVal = 0;
    CUDA_CHECK(cudaMemcpy(&minVal, ws.d_minInt, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&maxVal, ws.d_maxInt, sizeof(int), cudaMemcpyDeviceToHost));

    return {minVal, maxVal};
}

// -------------------------------------------------------------------------
// Initialize temperature
// -------------------------------------------------------------------------
__global__ void initializeTemperatureFieldKernel(
    double* __restrict__ temperature, const int* __restrict__ weightField,
    std::size_t totalCells, std::size_t width, std::size_t height,
    double x0, double y0, double dx, double dy,
    double meanDiscrepancy, int minWeight, int maxWeight)
{
    const double denom =
        (maxWeight > minWeight) ? static_cast<double>(maxWeight - minWeight) : 1.0;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalCells; idx += gridDim.x * blockDim.x)
    {
        const std::size_t i = idx % width;
        const std::size_t j = idx / width;

        const double x = x0 + dx * static_cast<double>(i);
        const double y = y0 + dy * static_cast<double>(j);

        const double normalizedWeight =
            static_cast<double>(weightField[idx] - minWeight) / denom;

        temperature[idx] =
            293.16 + 80.0 *
            (meanDiscrepancy + analyticalReferenceField(x, y)) *
            normalizedWeight;
    }
}

void initializeTemperatureFieldCUDA(
    double* d_temperature, const int* d_weightField,
    const SimulationConfig& cfg, const GridMapping& mapping,
    double meanDiscrepancy, int minWeight, int maxWeight)
{
    const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);
    const int block = 256;
    const int grid = static_cast<int>(
        std::min<std::size_t>((totalCells + block - 1) / block, 4096));

    initializeTemperatureFieldKernel<<<grid, block>>>(
        d_temperature, d_weightField, totalCells, cfg.gridWidth, cfg.gridHeight,
        mapping.x0, mapping.y0, mapping.dx, mapping.dy,
        meanDiscrepancy, minWeight, maxWeight);

    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------------------------
// Advance temperature field
// -------------------------------------------------------------------------
__global__ void updateInteriorKernel(
    const double* __restrict__ current, double* __restrict__ next,
    std::size_t width, std::size_t height, UpdateCoefficients coeffs)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const std::size_t j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= width - 1 || j >= height - 1) return;

    const std::size_t idx = linearIndex(i, j, width);

    next[idx] =
        coeffs.coeffX * (
            current[linearIndex(i - 1, j, width)] +
            current[linearIndex(i + 1, j, width)] +
            (coeffs.laplaceX + 0.5 / coeffs.coeffX) * current[idx]
        )
        +
        coeffs.coeffY * (
            current[linearIndex(i, j - 1, width)] +
            current[linearIndex(i, j + 1, width)] +
            (coeffs.laplaceY + 0.5 / coeffs.coeffY) * current[idx]
        );
}

__global__ void applyLeftRightBoundaryKernel(
    double* field, std::size_t width, std::size_t height)
{
    const std::size_t total = 2 * (height - 2);
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;    

    const std::size_t i = (idx % 2 == 0) ? 0 : width - 1; // 0 => left, 1 => right
    const std::size_t j = idx / 2 + 1; // 1 .. height-2

    field[linearIndex(i, j, width)] =
        field[linearIndex((i == 0) ? 1 : width - 2, j, width)];
}

__global__ void applyTopBottomBoundaryKernel(
    double* field, std::size_t width, std::size_t height)
{
    const std::size_t total = 2 * width;
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;    

    const std::size_t i = idx / 2;           // 0 .. width-1
    const std::size_t j = (idx % 2 == 0) ? 0 : height - 1; // 0 => top, 1 => bottom

    field[linearIndex(i, j, width)] =
        field[linearIndex(i, (j == 0) ? 1 : height - 2, width)];
}

void advanceTemperatureFieldCUDA(
    const double* d_current, double* d_next,
    std::size_t width, std::size_t height, const UpdateCoefficients& coeffs)
{
    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned>((width - 2 + block.x - 1) / block.x),
        static_cast<unsigned>((height - 2 + block.y - 1) / block.y));

    updateInteriorKernel<<<grid, block>>>(d_current, d_next, width, height, coeffs);
    CUDA_CHECK(cudaGetLastError());

    const int threads = 256;
    const std::size_t leftRightCells = 2 * (height - 2);
    const int leftRightBlocks = static_cast<int>(
        (leftRightCells + threads - 1) / threads);
    applyLeftRightBoundaryKernel<<<leftRightBlocks, threads>>>(
        d_next, width, height);
    CUDA_CHECK(cudaGetLastError());

    const std::size_t topBottomCells = 2 * width;
    const int topBottomBlocks = static_cast<int>(
        (topBottomCells + threads - 1) / threads);
    applyTopBottomBoundaryKernel<<<topBottomBlocks, threads>>>(
        d_next, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------------------------
// Compute Field Statistics
// -------------------------------------------------------------------------
__global__ void statsPass1PartialKernel(
    const double* __restrict__ field, std::size_t n,
    double* __restrict__ blockMin, double* __restrict__ blockMax,
    double* __restrict__ blockSum, double* __restrict__ blockSumSq,
    double* __restrict__ blockChecksum)
{
    extern __shared__ double smem_partial_kernel[];
    double* sMin      = smem_partial_kernel;
    double* sMax      = smem_partial_kernel + blockDim.x;
    double* sSum      = smem_partial_kernel + 2 * blockDim.x;
    double* sSumSq    = smem_partial_kernel + 3 * blockDim.x;
    double* sChecksum = smem_partial_kernel + 4 * blockDim.x;

    const int t = threadIdx.x;

    double localMin = INFINITY;
    double localMax = -INFINITY;
    double localSum = 0.0;
    double localSumSq = 0.0;
    double localChecksum = 0.0;

    for (std::size_t i = blockIdx.x * blockDim.x + t;
         i < n; i += gridDim.x * blockDim.x)
    {
        const double v = field[i];
        localMin = fmin(localMin, v);
        localMax = fmax(localMax, v);
        localSum += v;
        localSumSq += v * v;
        localChecksum += v * static_cast<double>((i % 1009ULL) + 1ULL);
    }

    sMin[t] = localMin;
    sMax[t] = localMax;
    sSum[t] = localSum;
    sSumSq[t] = localSumSq;
    sChecksum[t] = localChecksum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sMin[t] = fmin(sMin[t], sMin[t + stride]);
            sMax[t] = fmax(sMax[t], sMax[t + stride]);
            sSum[t] += sSum[t + stride];
            sSumSq[t] += sSumSq[t + stride];
            sChecksum[t] += sChecksum[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        blockMin[blockIdx.x] = sMin[0];
        blockMax[blockIdx.x] = sMax[0];
        blockSum[blockIdx.x] = sSum[0];
        blockSumSq[blockIdx.x] = sSumSq[0];
        blockChecksum[blockIdx.x] = sChecksum[0];
    }
}

__global__ void statsPass1FinalKernel(
    const double* __restrict__ blockMin, const double* __restrict__ blockMax,
    const double* __restrict__ blockSum, const double* __restrict__ blockSumSq,
    const double* __restrict__ blockChecksum, int numBlocks,
    double* __restrict__ outMin, double* __restrict__ outMax,
    double* __restrict__ outSum, double* __restrict__ outSumSq,
    double* __restrict__ outChecksum)
{
    extern __shared__ double smem_final_kernel[];
    double* sMin      = smem_final_kernel;
    double* sMax      = smem_final_kernel + blockDim.x;
    double* sSum      = smem_final_kernel + 2 * blockDim.x;
    double* sSumSq    = smem_final_kernel + 3 * blockDim.x;
    double* sChecksum = smem_final_kernel + 4 * blockDim.x;

    const int t = threadIdx.x;

    double localMin = INFINITY;
    double localMax = -INFINITY;
    double localSum = 0.0;
    double localSumSq = 0.0;
    double localChecksum = 0.0;

    for (int i = t; i < numBlocks; i += blockDim.x) {
        localMin = fmin(localMin, blockMin[i]);
        localMax = fmax(localMax, blockMax[i]);
        localSum += blockSum[i];
        localSumSq += blockSumSq[i];
        localChecksum += blockChecksum[i];
    }

    sMin[t] = localMin;
    sMax[t] = localMax;
    sSum[t] = localSum;
    sSumSq[t] = localSumSq;
    sChecksum[t] = localChecksum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sMin[t] = fmin(sMin[t], sMin[t + stride]);
            sMax[t] = fmax(sMax[t], sMax[t + stride]);
            sSum[t] += sSum[t + stride];
            sSumSq[t] += sSumSq[t + stride];
            sChecksum[t] += sChecksum[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        *outMin = sMin[0];
        *outMax = sMax[0];
        *outSum = sSum[0];
        *outSumSq = sSumSq[0];
        *outChecksum = sChecksum[0];
    }
}

__global__ void statsPass2PartialKernel(
    const double* __restrict__ field, std::size_t n, double mean,
    double* __restrict__ blockSumSqDiff)
{
    extern __shared__ double smem_2_partial_kernel[];
    double* s = smem_2_partial_kernel;
    const int t = threadIdx.x;

    double local = 0.0;

    for (std::size_t i = blockIdx.x * blockDim.x + t;
         i < n; i += gridDim.x * blockDim.x)
    {
        const double diff = field[i] - mean;
        local += diff * diff;
    }

    s[t] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }

    if (t == 0) blockSumSqDiff[blockIdx.x] = s[0];
}

__global__ void statsPass2FinalKernel(
    const double* __restrict__ blockSumSqDiff, int numBlocks,
    double* __restrict__ outSumSqDiff)
{
    extern __shared__ double smem_2_final_kernel[];
    double* s = smem_2_final_kernel;
    const int t = threadIdx.x;

    double local = 0.0;
    for (int i = t; i < numBlocks; i += blockDim.x) {
        local += blockSumSqDiff[i];
    }

    s[t] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }

    if (t == 0) *outSumSqDiff = s[0];
}

FieldStatistics computeFieldStatisticsCUDA(
    const double* d_field, std::size_t n, const DeviceWorkspace& ws)
{
    if (n == 0) {
        throw std::runtime_error("computeFieldStatisticsCUDA: empty field");
    }

    const int block = 256;
    const std::size_t shared1 = 5 * block * sizeof(double);

    statsPass1PartialKernel<<<ws.grid_size, block, shared1>>>(
        d_field, n,
        ws.d_blockMin, ws.d_blockMax,
        ws.d_blockSum, ws.d_blockSumSq, ws.d_blockChecksum);
    CUDA_CHECK(cudaGetLastError());

    statsPass1FinalKernel<<<1, block, shared1>>>(
        ws.d_blockMin, ws.d_blockMax, ws.d_blockSum, ws.d_blockSumSq, ws.d_blockChecksum,
        ws.grid_size,
        ws.d_min, ws.d_max, ws.d_sum, ws.d_sumSq, ws.d_checksum);
    CUDA_CHECK(cudaGetLastError());

    double hMin = 0.0;
    double hMax = 0.0;
    double hSum = 0.0;
    double hSumSq = 0.0;
    double hChecksum = 0.0;

    CUDA_CHECK(cudaMemcpy(&hMin, ws.d_min, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hMax, ws.d_max, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hSum, ws.d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hSumSq, ws.d_sumSq, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hChecksum, ws.d_checksum, sizeof(double), cudaMemcpyDeviceToHost));

    const double mean = hSum / static_cast<double>(n);

    statsPass2PartialKernel<<<ws.grid_size, block, block * sizeof(double)>>>(
        d_field, n, mean, ws.d_blockSqDiff);
    CUDA_CHECK(cudaGetLastError());

    statsPass2FinalKernel<<<1, block, block * sizeof(double)>>>(
        ws.d_blockSqDiff, ws.grid_size, ws.d_sqDiff);
    CUDA_CHECK(cudaGetLastError());

    double hSqDiff = 0.0;
    CUDA_CHECK(cudaMemcpy(&hSqDiff, ws.d_sqDiff, sizeof(double), cudaMemcpyDeviceToHost));

    FieldStatistics stats{};
    stats.minValue = hMin;
    stats.meanValue = mean;
    stats.maxValue = hMax;
    stats.stdDev = std::sqrt(hSqDiff / static_cast<double>(n));
    stats.l2Norm = std::sqrt(hSumSq);
    stats.checksum = hChecksum;

    return stats;
}
