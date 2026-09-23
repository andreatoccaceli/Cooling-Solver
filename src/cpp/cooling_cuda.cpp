// cooling_cuda.cpp
//
// Host-side translation unit: file I/O, argument parsing, orchestration.
// All CUDA kernels and the DeviceWorkspace method bodies live in kernels.cu.
// Shared types and inline helpers live in cooling_types.hpp.

#ifdef USE_HDF5
#include <H5Cpp.h>
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "utils/common.hpp"

// -------------------------------------------------------------------------
// Host-side CUDA error checking
// -------------------------------------------------------------------------
#define CUDA_CHECK_HOST(call)                                                \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err__));             \
        }                                                                    \
    } while (0)

// -------------------------------------------------------------------------
// Forward declarations of CUDA-side entry points (defined in kernels.cu)
// -------------------------------------------------------------------------
void computeFractalWeightsCUDA(
    int* d_weightField,
    const SimulationConfig& cfg,
    const GridMapping& mapping);

std::pair<int, int> computeWeightRangeCUDA(
    const int* d_weightField,
    std::size_t totalCells,
    const DeviceWorkspace& ws);

void initializeTemperatureFieldCUDA(
    double* d_temperature,
    const int* d_weightField,
    const SimulationConfig& cfg,
    const GridMapping& mapping,
    double meanDiscrepancy,
    int minWeight,
    int maxWeight);

void advanceTemperatureFieldCUDA(
    const double* d_current,
    double* d_next,
    std::size_t width,
    std::size_t height,
    const UpdateCoefficients& coeffs);

FieldStatistics computeFieldStatisticsCUDA(
    const double* d_field,
    std::size_t n,
    const DeviceWorkspace& ws);

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
#ifdef USE_HDF5
    H5::Exception::dontPrint();
#endif

    try {
        const CommandLineOptions cli = parseCommandLineArguments(argc, argv);
        SimulationConfig cfg = readConfigurationFile(cli.inputFile);

        if (cli.overrideOutputEvery) {
            cfg.outputEvery = cli.outputEvery;
        }

#ifndef USE_HDF5
        if (cli.writeHdf5) {
            throw std::runtime_error(
                "HDF5 output requested, but executable was built without HDF5 support. "
                "Use 'none' for the HDF5 argument or rebuild with -DUSE_HDF5.");
        }
#endif

        ensureParentDirectoryExists(cli.csvFile);
        if (cli.writeHdf5) {
            ensureParentDirectoryExists(cli.h5File);
        }

        const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);
        const GridMapping mapping    = buildGridMapping(cfg);
        const UpdateCoefficients coeffs =
            buildUpdateCoefficients(mapping.dx, mapping.dy, 100.0);
        const double meanDiscrepancy = computeMeanDiscrepancy(cfg);

#ifdef USE_HDF5
        constexpr bool hdf5Compiled = true;
#else
        constexpr bool hdf5Compiled = false;
#endif

        printRunHeader(cli, cfg, hdf5Compiled);

        // Host staging buffer — only needed when HDF5 output is enabled.
        std::vector<double> hostCurrentField;
        if (cli.writeHdf5) {
            hostCurrentField.resize(totalCells);
        }

        std::ofstream csv(cli.csvFile);
        if (!csv) {
            throw std::runtime_error("Cannot open CSV output file: " + cli.csvFile);
        }
        writeStatisticsHeader(csv);

        // Device memory
        int*    d_weightField  = nullptr;
        double* d_currentField = nullptr;
        double* d_nextField    = nullptr;

        CUDA_CHECK_HOST(cudaMalloc((void**)&d_weightField,  totalCells * sizeof(int)));
        CUDA_CHECK_HOST(cudaMalloc((void**)&d_currentField, totalCells * sizeof(double)));
        CUDA_CHECK_HOST(cudaMalloc((void**)&d_nextField,    totalCells * sizeof(double)));

        // Reduction scratch, allocated once for the whole run.
        DeviceWorkspace ws;
        ws.allocate(totalCells);

        ScopedTimer totalTimer;

        // --- Initialization ---
        ScopedTimer weightTimer;
        computeFractalWeightsCUDA(d_weightField, cfg, mapping);
        CUDA_CHECK_HOST(cudaDeviceSynchronize());
        const double weightTime = weightTimer.elapsedSeconds();

        ScopedTimer rangeTimer;
        const auto [minWeight, maxWeight] =
            computeWeightRangeCUDA(d_weightField, totalCells, ws);
        const double weightRangeTime = rangeTimer.elapsedSeconds();

        ScopedTimer initTimer;
        initializeTemperatureFieldCUDA(
            d_currentField, d_weightField, cfg, mapping,
            meanDiscrepancy, minWeight, maxWeight);
        CUDA_CHECK_HOST(cudaDeviceSynchronize());
        const double initTime = initTimer.elapsedSeconds();

        std::unique_ptr<TimeSeriesWriter> writer;
        if (cli.writeHdf5) {
            writer = std::make_unique<TimeSeriesWriter>(
                cli.h5File, cfg.gridWidth, cfg.gridHeight, 32, 256, 256);
        }

        double pureDynamicsTime = 0.0;
        double statisticsTime   = 0.0;
        double csvTime          = 0.0;
        double hdf5Time         = 0.0;

        int  outputFrames       = 0;
        bool hasLastWrittenStep = false;
        int  lastWrittenStep    = -1;
        FieldStatistics finalStats{};

        auto writeOutputFrame = [&](int step) {
            if (hasLastWrittenStep && step == lastWrittenStep) {
                return;
            }

            ScopedTimer statsTimer;
            const FieldStatistics stats =
                computeFieldStatisticsCUDA(d_currentField, totalCells, ws);
            CUDA_CHECK_HOST(cudaDeviceSynchronize());
            statisticsTime += statsTimer.elapsedSeconds();
            finalStats = stats;

            ScopedTimer csvTimer;
            writeStatisticsRow(csv, step, stats);
            csvTime += csvTimer.elapsedSeconds();

            if (writer) {
                ScopedTimer hdf5Timer;
                CUDA_CHECK_HOST(cudaMemcpy(
                    hostCurrentField.data(),
                    d_currentField,
                    totalCells * sizeof(double),
                    cudaMemcpyDeviceToHost));
                writer->writeFrame(step, hostCurrentField);
                hdf5Time += hdf5Timer.elapsedSeconds();
            }

            ++outputFrames;
            hasLastWrittenStep = true;
            lastWrittenStep = step;
        };

        ScopedTimer loopTimer;

        if (shouldWriteStep(0, cfg.timeSteps, cfg.outputEvery)) {
            writeOutputFrame(0);
        }

        for (int step = 1; step <= cfg.timeSteps; ++step) {
            ScopedTimer stepTimer;

            advanceTemperatureFieldCUDA(
                d_currentField, d_nextField,
                cfg.gridWidth, cfg.gridHeight, coeffs);

            CUDA_CHECK_HOST(cudaDeviceSynchronize());
            pureDynamicsTime += stepTimer.elapsedSeconds();

            std::swap(d_currentField, d_nextField);

            if (shouldWriteStep(step, cfg.timeSteps, cfg.outputEvery)) {
                writeOutputFrame(step);
            }
        }

        if (writer) {
            writer->close();
        }
        csv.flush();

        const double loopWallTime = loopTimer.elapsedSeconds();
        const double totalWallTime = totalTimer.elapsedSeconds();
        const double interiorWidth  = static_cast<double>(cfg.gridWidth  - 2);
        const double interiorHeight = static_cast<double>(cfg.gridHeight - 2);
        const double steps          = static_cast<double>(cfg.timeSteps);
        const double updates        = interiorWidth * interiorHeight * steps;

        // Explicit cleanup — ws destructor would also handle this, but doing
        // it here keeps the success path symmetrical with the failure path.
        ws.release();
        CUDA_CHECK_HOST(cudaFree(d_weightField));
        CUDA_CHECK_HOST(cudaFree(d_currentField));
        CUDA_CHECK_HOST(cudaFree(d_nextField));

        std::cout << "Weight field time:             " << weightTime       << " s\n";
        std::cout << "Weight range reduction time:   " << weightRangeTime  << " s\n";
        std::cout << "Initialization time:           " << initTime         << " s\n";
        std::cout << "Pure dynamics compute time:    " << pureDynamicsTime << " s\n";
        std::cout << "Statistics time:               " << statisticsTime   << " s\n";
        std::cout << "CSV write time:                " << csvTime          << " s\n";
        std::cout << "HDF5 write time:               " << hdf5Time         << " s\n";
        std::cout << "Dynamics loop wall time:       " << loopWallTime     << " s\n";
        std::cout << "Total measured wall time:      " << totalWallTime    << " s\n";
        std::cout << "Output frames:                 " << outputFrames     << '\n';

        if (cfg.timeSteps > 0 && pureDynamicsTime > 0.0) {
            std::cout << "Pure dynamics performance:     "
                      << updates / pureDynamicsTime / 1.0e9 << " GLUP/s\n";
        }
        if (cfg.timeSteps > 0 && loopWallTime > 0.0) {
            std::cout << "Loop end-to-end performance:   "
                      << updates / loopWallTime / 1.0e9 << " GLUP/s\n";
        }

        std::cout << "Mean discrepancy:              "
                  << std::setprecision(15) << meanDiscrepancy      << '\n';
        std::cout << "Final min:                     "
                  << std::setprecision(15) << finalStats.minValue  << '\n';
        std::cout << "Final mean:                    "
                  << std::setprecision(15) << finalStats.meanValue << '\n';
        std::cout << "Final max:                     "
                  << std::setprecision(15) << finalStats.maxValue  << '\n';
        std::cout << "Final std.dev.:                "
                  << std::setprecision(15) << finalStats.stdDev    << '\n';
        std::cout << "Final L2 norm:                 "
                  << std::setprecision(15) << finalStats.l2Norm    << '\n';
        std::cout << "Final checksum:                "
                  << std::setprecision(15) << finalStats.checksum  << '\n';
        std::cout << "Weight range:                  "
                  << minWeight << " ... " << maxWeight              << '\n';
        std::cout << "\nSimulation completed successfully.\n";

        return 0;

#ifdef USE_HDF5
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 ERROR: " << e.getDetailMsg() << '\n';
        return 1;
#endif
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "CRITICAL ERROR: unknown failure\n";
        return 1;
    }
}
