#pragma once

#ifdef USE_HDF5
#include <H5Cpp.h>
#endif

// -------------------------------------------------------------------------
// Shared types and helpers used by both the host TU (cooling_cuda.cpp)
// and the CUDA TU (kernels.cu). Compiles under g++/clang and nvcc.
// -------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using index_t = std::ptrdiff_t;

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

struct ScopedTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point start{clock::now()};

    double elapsedSeconds() const {
        return std::chrono::duration<double>(clock::now() - start).count();
    }
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

#ifdef USE_HDF5

class TimeSeriesWriter {
public:
    TimeSeriesWriter(
        const std::string& fileName,
        std::size_t width,
        std::size_t height,
        std::size_t batch = 32,
        std::size_t tileY = 256,
        std::size_t tileX = 256
    )
        : file_(fileName, H5F_ACC_TRUNC),
          width_(width),
          height_(height),
          batch_(batch),
          frameCount_(0),
          capacity_(batch),
          closed_(false)
    {
        if (width_ == 0 || height_ == 0) {
            throw std::invalid_argument("TimeSeriesWriter: invalid dimensions");
        }

        if (batch_ == 0) {
            throw std::invalid_argument("TimeSeriesWriter: batch must be positive");
        }

        const hsize_t chunkY = static_cast<hsize_t>(std::min<std::size_t>(height_, tileY));
        const hsize_t chunkX = static_cast<hsize_t>(std::min<std::size_t>(width_, tileX));

        {
            hsize_t dims[3] = {0, static_cast<hsize_t>(height_), static_cast<hsize_t>(width_)};
            hsize_t maxdims[3] = {H5S_UNLIMITED, static_cast<hsize_t>(height_), static_cast<hsize_t>(width_)};

            H5::DataSpace space(3, dims, maxdims);
            H5::DSetCreatPropList props;

            hsize_t chunks[3] = {1, chunkY, chunkX};
            props.setChunk(3, chunks);

            fieldDataset_ = file_.createDataSet(
                "/field",
                H5::PredType::NATIVE_DOUBLE,
                space,
                props
            );
        }

        {
            hsize_t dims[1] = {0};
            hsize_t maxdims[1] = {H5S_UNLIMITED};

            H5::DataSpace space(1, dims, maxdims);
            H5::DSetCreatPropList props;

            hsize_t chunks[1] = {static_cast<hsize_t>(batch_)};
            props.setChunk(1, chunks);

            stepDataset_ = file_.createDataSet(
                "/step",
                H5::PredType::NATIVE_INT,
                space,
                props
            );
        }

        extend(capacity_);
    }

    ~TimeSeriesWriter() {
        try {
            close();
        } catch (...) {
        }
    }

    TimeSeriesWriter(const TimeSeriesWriter&) = delete;
    TimeSeriesWriter& operator=(const TimeSeriesWriter&) = delete;

    void writeFrame(int stepNumber, const std::vector<double>& field) {
        if (closed_) {
            throw std::runtime_error("TimeSeriesWriter: write after close");
        }

        if (field.size() != checkedGridSize(width_, height_)) {
            throw std::runtime_error("TimeSeriesWriter: field size mismatch");
        }

        if (frameCount_ >= capacity_) {
            capacity_ += batch_;
            extend(capacity_);
        }

        {
            H5::DataSpace filespace = fieldDataset_.getSpace();

            hsize_t start[3] = {static_cast<hsize_t>(frameCount_), 0, 0};
            hsize_t count[3] = {
                1,
                static_cast<hsize_t>(height_),
                static_cast<hsize_t>(width_)
            };

            filespace.selectHyperslab(H5S_SELECT_SET, count, start);

            H5::DataSpace memspace(3, count);

            fieldDataset_.write(
                field.data(),
                H5::PredType::NATIVE_DOUBLE,
                memspace,
                filespace
            );
        }

        {
            H5::DataSpace filespace = stepDataset_.getSpace();

            hsize_t start[1] = {static_cast<hsize_t>(frameCount_)};
            hsize_t count[1] = {1};

            filespace.selectHyperslab(H5S_SELECT_SET, count, start);

            H5::DataSpace memspace(1, count);

            int value = stepNumber;

            stepDataset_.write(
                &value,
                H5::PredType::NATIVE_INT,
                memspace,
                filespace
            );
        }

        ++frameCount_;
    }

    void close() {
        if (closed_) {
            return;
        }

        if (frameCount_ != capacity_) {
            extend(frameCount_);
        }

        file_.flush(H5F_SCOPE_GLOBAL);
        fieldDataset_.close();
        stepDataset_.close();
        file_.close();
        closed_ = true;
    }

private:
    void extend(std::size_t newSize) {
        hsize_t fieldDims[3] = {
            static_cast<hsize_t>(newSize),
            static_cast<hsize_t>(height_),
            static_cast<hsize_t>(width_)
        };

        fieldDataset_.extend(fieldDims);

        hsize_t stepDims[1] = {static_cast<hsize_t>(newSize)};
        stepDataset_.extend(stepDims);
    }

    H5::H5File file_;
    H5::DataSet fieldDataset_;
    H5::DataSet stepDataset_;
    std::size_t width_{};
    std::size_t height_{};
    std::size_t batch_{};
    std::size_t frameCount_{};
    std::size_t capacity_{};
    bool closed_{false};
};

#else

class TimeSeriesWriter {
public:
    TimeSeriesWriter(
        const std::string&,
        std::size_t,
        std::size_t,
        std::size_t = 32,
        std::size_t = 256,
        std::size_t = 256
    )
    {
        throw std::runtime_error(
            "This executable was built without HDF5 support. "
            "Recompile with -DUSE_HDF5 to enable HDF5 output."
        );
    }

    void writeFrame(int, const std::vector<double>&) {}
    void close() {}
};

#endif
