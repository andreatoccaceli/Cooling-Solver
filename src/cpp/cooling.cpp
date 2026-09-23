/*
================================================================================
Cooling Field Solver - Serial C++17 Baseline
================================================================================

Course final project baseline.

This program is a serial mini-application for parallelization and performance
analysis. Students may develop a parallel version using one or more of:

  - OpenMP
  - MPI
  - CUDA
  - OpenACC
  - hybrid approaches

The application contains:
  - an irregular field-weight computation,
  - a field initialization phase,
  - a 2D iterative stencil update,
  - global statistical reductions,
  - optional HDF5 output.

Performance/Benchmarking mode: HDF5 output disabled.

Recommended benchmarking run style:

  ./path/to/cooling_serial input/Cooling.in none output/Cooling_cpp.csv 0

HDF5 support: HDF5 is optional at compile time.

  Without HDF5:

  g++ -O3 -std=c++17 -Wall -Wextra -pedantic cooling.cpp -o cooling_serial

  With HDF5:

  g++ -O3 -std=c++17 -Wall -Wextra -pedantic -DUSE_HDF5 cooling.cpp -o cooling_serial -lhdf5_cpp -lhdf5

  Depending on the cluster configuration, students may need an HDF5 compiler
  wrapper, modules, or explicit include/library paths.

Input file format, after removing comments beginning with '#':

  gridWidth
  gridHeight
  numberOfMeasuredPoints
  x y value        repeated numberOfMeasuredPoints times
  domainStartX
  domainStartY
  domainWidth
  domainHeight
  maxFractalIterations
  timeSteps
  outputEvery      optional; 0 means final step only

Command line:

  ./path/to/cooling_serial [inputFile] [h5File|none|--no-hdf5] [csvFile] [outputEvery]

Examples:

  ./path/to/cooling_serial input/Cooling.in
  ./path/to/cooling_serial input/Cooling.in none                  output/Cooling_cpp.csv  0
  ./path/to/cooling_serial input/Cooling.in output/Cooling_cpp.h5 output/Cooling_cpp.csv 50

================================================================================
*/

#ifdef USE_HDF5
#include <H5Cpp.h>
#endif

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

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#include "utils/common.hpp"

void computeFractalWeights(
    std::vector<int>& weightField,
    const SimulationConfig& cfg,
    const GridMapping& mapping
) {
    const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);

    if (weightField.size() != totalCells) {
        throw std::runtime_error("computeFractalWeights: size mismatch");
    }

    const index_t width = static_cast<index_t>(cfg.gridWidth);
    const index_t height = static_cast<index_t>(cfg.gridHeight);

    for (index_t j = 0; j < height; ++j) {
        for (index_t i = 0; i < width; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t jj = static_cast<std::size_t>(j);
            const std::size_t idx = linearIndex(ii, jj, cfg.gridWidth);

            const double cReal = mapping.x0 + mapping.dx * static_cast<double>(i);
            const double cImag = mapping.y0 + mapping.dy * static_cast<double>(j);

            double zReal = 0.0;
            double zImag = 0.0;
            int iter = 0;

            for (; iter < cfg.maxFractalIterations; ++iter) {
                if (zReal * zReal + zImag * zImag > 4.0) {
                    break;
                }

                const double tmp = zReal * zReal - zImag * zImag + cReal;
                zImag = 2.0 * zReal * zImag + cImag;
                zReal = tmp;
            }

            weightField[idx] = iter;
        }
    }
}

std::pair<int, int> computeWeightRange(const std::vector<int>& weightField) {
    if (weightField.empty()) {
        throw std::runtime_error("computeWeightRange: empty field");
    }

    const auto [minIt, maxIt] = std::minmax_element(weightField.begin(), weightField.end());
    return {*minIt, *maxIt};
}

void initializeTemperatureField(
    std::vector<double>& temperature,
    const std::vector<int>& weightField,
    const SimulationConfig& cfg,
    const GridMapping& mapping,
    double meanDiscrepancy,
    int minWeight,
    int maxWeight
) {
    const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);

    if (temperature.size() != totalCells || weightField.size() != totalCells) {
        throw std::runtime_error("initializeTemperatureField: size mismatch");
    }

    const double denom = (maxWeight > minWeight) ? static_cast<double>(maxWeight - minWeight) : 1.0;

    const index_t width = static_cast<index_t>(cfg.gridWidth);
    const index_t height = static_cast<index_t>(cfg.gridHeight);

    for (index_t j = 0; j < height; ++j) {
        for (index_t i = 0; i < width; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t jj = static_cast<std::size_t>(j);
            const std::size_t idx = linearIndex(ii, jj, cfg.gridWidth);

            const double x = mapping.x0 + mapping.dx * static_cast<double>(i);
            const double y = mapping.y0 + mapping.dy * static_cast<double>(j);

            const double normalizedWeight = static_cast<double>(weightField[idx] - minWeight) / denom;

            temperature[idx] = 293.16 + 80.0 * (meanDiscrepancy + analyticalReferenceField(x, y)) * normalizedWeight;
        }
    }
}

void updateInterior(
    const double* RESTRICT current,
    double* RESTRICT next,
    std::size_t width,
    std::size_t height,
    const UpdateCoefficients& coeffs
) {
    const index_t w = static_cast<index_t>(width);
    const index_t h = static_cast<index_t>(height);

    for (index_t j = 1; j < h - 1; ++j) {
        for (index_t i = 1; i < w - 1; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t jj = static_cast<std::size_t>(j);
            const std::size_t idx = linearIndex(ii, jj, width);

            next[idx] =
                coeffs.coeffX * (
                    current[linearIndex(ii - 1, jj, width)] +
                    current[linearIndex(ii + 1, jj, width)] +
                    (coeffs.laplaceX + 0.5 / coeffs.coeffX) * current[idx]
                )
                +
                coeffs.coeffY * (
                    current[linearIndex(ii, jj - 1, width)] +
                    current[linearIndex(ii, jj + 1, width)] +
                    (coeffs.laplaceY + 0.5 / coeffs.coeffY) * current[idx]
                );
        }
    }
}

void applyBoundaryConditions(double* field, std::size_t width, std::size_t height) {
    for (std::size_t j = 1; j < height - 1; ++j) {
        field[linearIndex(0, j, width)] = field[linearIndex(1, j, width)];
        field[linearIndex(width - 1, j, width)] = field[linearIndex(width - 2, j, width)];
    }

    for (std::size_t i = 0; i < width; ++i) {
        field[linearIndex(i, 0, width)] = field[linearIndex(i, 1, width)];
        field[linearIndex(i, height - 1, width)] = field[linearIndex(i, height - 2, width)];
    }
}

void advanceTemperatureField(
    const double* current,
    double* next,
    std::size_t width,
    std::size_t height,
    const UpdateCoefficients& coeffs
) {
    updateInterior(current, next, width, height, coeffs);
    applyBoundaryConditions(next, width, height);
}

FieldStatistics computeFieldStatistics(const std::vector<double>& field) {
    if (field.empty()) {
        throw std::runtime_error("computeFieldStatistics: empty field");
    }

    const std::size_t n = field.size();
    double minValue =  std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sumSquares = 0.0;
    double checksum = 0.0; // checksum is not a physical quantity. It is meant to help detect numerical differences between implementations.

    for (std::size_t i = 0; i < n; ++i) {
        const double value = field[i];
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        sum += value;
        sumSquares += value * value;
        checksum += value * static_cast<double>((i % 1009U) + 1U);
    }

    const double mean = sum / static_cast<double>(n);

    double sumSquaredDiff = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double diff = field[i] - mean;
        sumSquaredDiff += diff * diff;
    }

    FieldStatistics stats{};
    stats.minValue = minValue;
    stats.meanValue = mean;
    stats.maxValue = maxValue;
    stats.stdDev = std::sqrt(sumSquaredDiff / static_cast<double>(n));
    stats.l2Norm = std::sqrt(sumSquares);
    stats.checksum = checksum;

    return stats;
}

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
                "Use 'none' for the HDF5 argument or rebuild with -DUSE_HDF5."
            );
        }
#endif

        ensureParentDirectoryExists(cli.csvFile);

        if (cli.writeHdf5) {
            ensureParentDirectoryExists(cli.h5File);
        }

        const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);
        const GridMapping mapping = buildGridMapping(cfg);
        const UpdateCoefficients coeffs = buildUpdateCoefficients(mapping.dx, mapping.dy, 100.0);
        const double meanDiscrepancy = computeMeanDiscrepancy(cfg);

#ifdef USE_HDF5
        constexpr bool hdf5Compiled = true;
#else
        constexpr bool hdf5Compiled = false;
#endif

        printRunHeader(cli, cfg, hdf5Compiled);

        std::vector<int> weightField(totalCells);
        std::vector<double> currentField(totalCells);
        std::vector<double> nextField(totalCells);

        std::ofstream csv(cli.csvFile);

        if (!csv) {
            throw std::runtime_error("Cannot open CSV output file: " + cli.csvFile);
        }

        writeStatisticsHeader(csv);

        ScopedTimer totalTimer;

        ScopedTimer weightTimer;
        computeFractalWeights(weightField, cfg, mapping);
        const double weightTime = weightTimer.elapsedSeconds();

        ScopedTimer rangeTimer;
        const auto [minWeight, maxWeight] = computeWeightRange(weightField);
        const double weightRangeTime = rangeTimer.elapsedSeconds();

        ScopedTimer initTimer;
        initializeTemperatureField(
            currentField,
            weightField,
            cfg,
            mapping,
            meanDiscrepancy,
            minWeight,
            maxWeight
        );
        const double initTime = initTimer.elapsedSeconds();

        std::unique_ptr<TimeSeriesWriter> writer;

        if (cli.writeHdf5) {
            writer = std::make_unique<TimeSeriesWriter>(
                cli.h5File,
                cfg.gridWidth,
                cfg.gridHeight,
                32,
                256,
                256
            );
        }

        double pureDynamicsTime = 0.0;
        double statisticsTime   = 0.0;
        double csvTime  = 0.0;
        double hdf5Time = 0.0;

        int outputFrames = 0;
        bool hasLastWrittenStep = false;
        int lastWrittenStep = -1;

        FieldStatistics finalStats{};

        auto writeOutputFrame = [&](int step) {
            if (hasLastWrittenStep && step == lastWrittenStep) {
                return;
            }

            ScopedTimer statsTimer;
            const FieldStatistics stats = computeFieldStatistics(currentField);
            statisticsTime += statsTimer.elapsedSeconds();
            finalStats = stats;

            ScopedTimer csvTimer;
            writeStatisticsRow(csv, step, stats);
            csvTime += csvTimer.elapsedSeconds();

            if (writer) {
                ScopedTimer hdf5Timer;
                writer->writeFrame(step, currentField);
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
            advanceTemperatureField(
                currentField.data(),
                nextField.data(),
                cfg.gridWidth,
                cfg.gridHeight,
                coeffs
            );
            pureDynamicsTime += stepTimer.elapsedSeconds();

            std::swap(currentField, nextField);

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
        const double updates = interiorWidth * interiorHeight * steps;

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
            std::cout << "Pure dynamics performance:     " << updates / pureDynamicsTime / 1.0e9 << " GLUP/s\n";
        }

        if (cfg.timeSteps > 0 && loopWallTime > 0.0) {
            std::cout << "Loop end-to-end performance:   " << updates / loopWallTime / 1.0e9 << " GLUP/s\n";
        }

        std::cout << "Mean discrepancy:              " << std::setprecision(15) << meanDiscrepancy      << '\n';
        std::cout << "Final min:                     " << std::setprecision(15) << finalStats.minValue  << '\n';
        std::cout << "Final mean:                    " << std::setprecision(15) << finalStats.meanValue << '\n';
        std::cout << "Final max:                     " << std::setprecision(15) << finalStats.maxValue  << '\n';
        std::cout << "Final std.dev.:                " << std::setprecision(15) << finalStats.stdDev    << '\n';
        std::cout << "Final L2 norm:                 " << std::setprecision(15) << finalStats.l2Norm    << '\n';
        std::cout << "Final checksum:                " << std::setprecision(15) << finalStats.checksum  << '\n';
        std::cout << "Weight range:                  " << minWeight << " ... "  << maxWeight            << '\n';
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
