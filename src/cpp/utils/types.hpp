#pragma once

// -------------------------------------------------------------------------
// Shared types and helpers used by both the host TU (cooling_cuda.cpp)
// and the CUDA TU (kernels.cu). Compiles under g++/clang and nvcc.
// -------------------------------------------------------------------------

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <string>

// -------------------------------------------------------------------------
// Compiler plumbing: HD = __host__ __device__ when compiled by nvcc,
// empty otherwise. Lets one inline definition serve both toolchains.
// -------------------------------------------------------------------------
#if defined(__CUDACC__)
#  define COOLING_HD __host__ __device__
#else
#  define COOLING_HD
#endif

// -------------------------------------------------------------------------
// CPU-only data structures
// -------------------------------------------------------------------------

struct SamplePoint {
    double x{};
    double y{};
    double value{};
};

struct SimulationConfig {
    std::size_t gridWidth{};
    std::size_t gridHeight{};
    double domainStartX{};
    double domainStartY{};
    double domainWidth{};
    double domainHeight{};
    int maxFractalIterations{};
    int timeSteps{};
    int outputEvery{0};
    std::vector<SamplePoint> measuredPoints;
};

struct GridMapping {
    double x0{};
    double y0{};
    double dx{};
    double dy{};
};

struct UpdateCoefficients {
    double damping{};
    double stepX{};
    double stepY{};
    double coeffX{};
    double coeffY{};
    double laplaceX{};
    double laplaceY{};
};

struct FieldStatistics {
    double minValue{};
    double meanValue{};
    double maxValue{};
    double stdDev{};
    double l2Norm{};
    double checksum{};
};

// -------------------------------------------------------------------------
// Host-side command line / utility structures
// -------------------------------------------------------------------------

struct CommandLineOptions {
    std::string inputFile{"input/Cooling.in"};
    std::string h5File{"none"};
    std::string csvFile{"output/Cooling_cpp.csv"};
    bool writeHdf5{false};
    bool overrideOutputEvery{false};
    int outputEvery{0};
};

// -------------------------------------------------------------------------
// DeviceWorkspace — pre-allocated reduction scratch buffers.
//
// The struct itself is defined here (so both TUs see a complete type),
// but the allocate/release bodies live in kernels.cu so all CUDA runtime
// calls stay in the CUDA TU.
// -------------------------------------------------------------------------
struct DeviceWorkspace {
    // Integer reduction scratch (weight range)
    int* d_blockMinInt{};
    int* d_blockMaxInt{};
    int* d_minInt{};
    int* d_maxInt{};

    // Double reduction scratch (field statistics, pass 1)
    double* d_blockMin{};
    double* d_blockMax{};
    double* d_blockSum{};
    double* d_blockSumSq{};
    double* d_blockChecksum{};
    double* d_min{};
    double* d_max{};
    double* d_sum{};
    double* d_sumSq{};
    double* d_checksum{};

    // Double reduction scratch (field statistics, pass 2 — variance)
    double* d_blockSqDiff{};
    double* d_sqDiff{};

    int grid_size{};

    // Defined in kernels.cu
    DeviceWorkspace();
    ~DeviceWorkspace();

    // Non-copyable (raw device pointers). Movable is OK but we don't need it.
    DeviceWorkspace(const DeviceWorkspace&) = delete;
    DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;

    void allocate(std::size_t n);
    void release();
};

// -------------------------------------------------------------------------
// Shared inline helpers (compiled by both toolchains)
// -------------------------------------------------------------------------

COOLING_HD inline std::size_t linearIndex(
    std::size_t i, std::size_t j, std::size_t width) noexcept {
    return i + j * width;
}

COOLING_HD inline double analyticalReferenceField(double x, double y) noexcept {
    return (x * x * x + y * y * y) / 6.0;
}

// checkedGridSize is host-only (uses exceptions and <limits>).
inline std::size_t checkedGridSize(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument("Grid dimensions must be greater than zero");
    }
    if (width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::overflow_error("Grid size overflow");
    }
    return width * height;
}