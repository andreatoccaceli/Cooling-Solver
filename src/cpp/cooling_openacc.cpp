/* Cooling Field Solver - OpenACC version. */

#include "utils/common.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif

void computeFractalWeightsOpenAcc(
    int* weight,
    std::size_t totalCells,
    const SimulationConfig& cfg,
    const GridMapping& mapping
) {
    (void)totalCells;
    const index_t width = static_cast<index_t>(cfg.gridWidth);
    const index_t height = static_cast<index_t>(cfg.gridHeight);
    const double x0 = mapping.x0;
    const double y0 = mapping.y0;
    const double dx = mapping.dx;
    const double dy = mapping.dy;
    const int maxIterations = cfg.maxFractalIterations;

#pragma acc parallel loop collapse(2) present(weight[0:totalCells])
    for (index_t j = 0; j < height; ++j) {
        for (index_t i = 0; i < width; ++i) {
            const std::size_t idx =
                static_cast<std::size_t>(i + j * width);
            const double cReal = x0 + dx * static_cast<double>(i);
            const double cImag = y0 + dy * static_cast<double>(j);

            double zReal = 0.0;
            double zImag = 0.0;
            int iter = 0;

            for (; iter < maxIterations; ++iter) {
                if (zReal * zReal + zImag * zImag > 4.0) {
                    break;
                }

                const double nextReal =
                    zReal * zReal - zImag * zImag + cReal;
                zImag = 2.0 * zReal * zImag + cImag;
                zReal = nextReal;
            }

            weight[idx] = iter;
        }
    }
}

std::pair<int, int> computeWeightRangeOpenAcc(
    const int* weight,
    std::size_t totalCells
) {
    int minWeight = std::numeric_limits<int>::max();
    int maxWeight = std::numeric_limits<int>::lowest();
    const index_t n = static_cast<index_t>(totalCells);

#pragma acc parallel loop present(weight[0:totalCells]) \
    reduction(min:minWeight) reduction(max:maxWeight)
    for (index_t idx = 0; idx < n; ++idx) {
        const int value = weight[static_cast<std::size_t>(idx)];
        minWeight = value < minWeight ? value : minWeight;
        maxWeight = value > maxWeight ? value : maxWeight;
    }

    return {minWeight, maxWeight};
}

void initializeTemperatureFieldOpenAcc(
    double* temperature,
    const int* weight,
    std::size_t totalCells,
    const SimulationConfig& cfg,
    const GridMapping& mapping,
    double meanDiscrepancy,
    int minWeight,
    int maxWeight
) {
    (void)totalCells;
    const index_t width = static_cast<index_t>(cfg.gridWidth);
    const index_t height = static_cast<index_t>(cfg.gridHeight);
    const double x0 = mapping.x0;
    const double y0 = mapping.y0;
    const double dx = mapping.dx;
    const double dy = mapping.dy;
    const double denominator = maxWeight > minWeight
        ? static_cast<double>(maxWeight - minWeight)
        : 1.0;

#pragma acc parallel loop collapse(2) \
    present(temperature[0:totalCells], weight[0:totalCells])
    for (index_t j = 0; j < height; ++j) {
        for (index_t i = 0; i < width; ++i) {
            const std::size_t idx =
                static_cast<std::size_t>(i + j * width);
            const double x = x0 + dx * static_cast<double>(i);
            const double y = y0 + dy * static_cast<double>(j);
            const double reference = (x * x * x + y * y * y) / 6.0;
            const double normalizedWeight =
                static_cast<double>(weight[idx] - minWeight) / denominator;

            temperature[idx] =
                293.16
                + 80.0 * (meanDiscrepancy + reference) * normalizedWeight;
        }
    }
}

void advanceTemperatureFieldOpenAcc(
    const double* current,
    double* next,
    std::size_t totalCells,
    std::size_t width,
    std::size_t height,
    const UpdateCoefficients& coeffs
) {
    (void)totalCells;
    const index_t w = static_cast<index_t>(width);
    const index_t h = static_cast<index_t>(height);
    const double coeffX = coeffs.coeffX;
    const double coeffY = coeffs.coeffY;
    const double centerX = coeffs.laplaceX + 0.5 / coeffX;
    const double centerY = coeffs.laplaceY + 0.5 / coeffY;

#pragma acc parallel loop collapse(2) \
    present(current[0:totalCells], next[0:totalCells])
    for (index_t j = 1; j < h - 1; ++j) {
        for (index_t i = 1; i < w - 1; ++i) {
            const std::size_t idx =
                static_cast<std::size_t>(i + j * w);

            next[idx] =
                coeffX * (
                    current[idx - 1]
                    + current[idx + 1]
                    + centerX * current[idx]
                )
                + coeffY * (
                    current[idx - width]
                    + current[idx + width]
                    + centerY * current[idx]
                );
        }
    }

    // Keep these as two separate kernels. The second one also writes the
    // corners and therefore must observe the left/right boundary update.
#pragma acc parallel loop present(next[0:totalCells])
    for (index_t j = 1; j < h - 1; ++j) {
        const std::size_t row = static_cast<std::size_t>(j) * width;
        next[row] = next[row + 1];
        next[row + width - 1] = next[row + width - 2];
    }

#pragma acc parallel loop present(next[0:totalCells])
    for (index_t i = 0; i < w; ++i) {
        const std::size_t column = static_cast<std::size_t>(i);
        next[column] = next[column + width];
        next[column + (height - 1) * width] =
            next[column + (height - 2) * width];
    }
}

FieldStatistics computeFieldStatisticsOpenAcc(
    const double* field,
    std::size_t totalCells
) {
    if (totalCells == 0) {
        throw std::runtime_error("computeFieldStatisticsOpenAcc: empty field");
    }

    const index_t n = static_cast<index_t>(totalCells);
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sumSquares = 0.0;
    double checksum = 0.0;

#pragma acc parallel loop present(field[0:totalCells]) \
    reduction(min:minValue) reduction(max:maxValue) \
    reduction(+:sum,sumSquares,checksum)
    for (index_t idx = 0; idx < n; ++idx) {
        const std::size_t i = static_cast<std::size_t>(idx);
        const double value = field[i];
        minValue = value < minValue ? value : minValue;
        maxValue = value > maxValue ? value : maxValue;
        sum += value;
        sumSquares += value * value;
        checksum += value * static_cast<double>((i % 1009U) + 1U);
    }

    const double mean = sum / static_cast<double>(totalCells);
    double sumSquaredDiff = 0.0;

#pragma acc parallel loop present(field[0:totalCells]) \
    reduction(+:sumSquaredDiff)
    for (index_t idx = 0; idx < n; ++idx) {
        const double diff =
            field[static_cast<std::size_t>(idx)] - mean;
        sumSquaredDiff += diff * diff;
    }

    FieldStatistics stats{};
    stats.minValue = minValue;
    stats.meanValue = mean;
    stats.maxValue = maxValue;
    stats.stdDev = std::sqrt(
        sumSquaredDiff / static_cast<double>(totalCells)
    );
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
                "HDF5 output requested, but this executable was built "
                "without HDF5 support. Use 'none' or rebuild with "
                "-DUSE_HDF5."
            );
        }
#endif

        ensureParentDirectoryExists(cli.csvFile);
        if (cli.writeHdf5) {
            ensureParentDirectoryExists(cli.h5File);
        }

        const std::size_t totalCells =
            checkedGridSize(cfg.gridWidth, cfg.gridHeight);
        const GridMapping mapping = buildGridMapping(cfg);
        const UpdateCoefficients coeffs =
            buildUpdateCoefficients(mapping.dx, mapping.dy, 100.0);
        const double meanDiscrepancy = computeMeanDiscrepancy(cfg);

#ifdef USE_HDF5
        constexpr bool hdf5Compiled = true;
#else
        constexpr bool hdf5Compiled = false;
#endif

        printRunHeader(cli, cfg, hdf5Compiled);
#ifdef _OPENACC
        std::cout << "OpenACC enabled:               yes (spec "
                  << _OPENACC << ")\n";
        acc_device_t dev_type = acc_get_device_type();
	if (dev_type == 2){ // Running on multicore CPU
        	std::cout << "Cpu threads:                   "
			          << std::getenv("ACC_NUM_CORES") << '\n';
	}
#else
        std::cout << "OpenACC enabled:               no; pragmas run serially\n";
#endif

        std::vector<int> weightStorage(totalCells);
        std::vector<double> fieldStorageA(totalCells);
        std::vector<double> fieldStorageB(totalCells);
        int* weight = weightStorage.data();
        double* fieldA = fieldStorageA.data();
        double* fieldB = fieldStorageB.data();
        double* current = fieldA;
        double* next = fieldB;

        std::ofstream csv(cli.csvFile);
        if (!csv) {
            throw std::runtime_error(
                "Cannot open CSV output file: " + cli.csvFile
            );
        }
        writeStatisticsHeader(csv);

        double weightTime = 0.0;
        double weightRangeTime = 0.0;
        double initTime = 0.0;
        double pureDynamicsTime = 0.0;
        double statisticsTime = 0.0;
        double csvTime = 0.0;
        double hdf5Time = 0.0;
        double loopWallTime = 0.0;
        double totalWallTime = 0.0;
        int outputFrames = 0;
        bool hasLastWrittenStep = false;
        int lastWrittenStep = -1;
        int minWeight = 0;
        int maxWeight = 0;
        FieldStatistics finalStats{};

#pragma acc data create(weight[0:totalCells], fieldA[0:totalCells], fieldB[0:totalCells])
        {
            // Match the serial timer: storage allocation/data-region setup is
            // outside the measured interval.
            ScopedTimer totalTimer;

            ScopedTimer weightTimer;
            computeFractalWeightsOpenAcc(
                weight, totalCells, cfg, mapping
            );
            weightTime = weightTimer.elapsedSeconds();

            ScopedTimer rangeTimer;
            const auto range =
                computeWeightRangeOpenAcc(weight, totalCells);
            minWeight = range.first;
            maxWeight = range.second;
            weightRangeTime = rangeTimer.elapsedSeconds();

            ScopedTimer initTimer;
            initializeTemperatureFieldOpenAcc(
                current,
                weight,
                totalCells,
                cfg,
                mapping,
                meanDiscrepancy,
                minWeight,
                maxWeight
            );
            initTime = initTimer.elapsedSeconds();

            std::unique_ptr<TimeSeriesWriter> writer;
            if (cli.writeHdf5) {
                writer = std::make_unique<TimeSeriesWriter>(
                    cli.h5File, cfg.gridWidth, cfg.gridHeight, 32, 256, 256
                );
            }

            auto writeOutputFrame = [&](int step) {
                if (hasLastWrittenStep && step == lastWrittenStep) return;

                ScopedTimer statsTimer;
                const FieldStatistics stats =
                    computeFieldStatisticsOpenAcc(current, totalCells);
                statisticsTime += statsTimer.elapsedSeconds();
                finalStats = stats;

                ScopedTimer csvTimer;
                writeStatisticsRow(csv, step, stats);
                csvTime += csvTimer.elapsedSeconds();

                if (writer) {
                    ScopedTimer hdf5Timer;
#ifdef _OPENACC
                    const std::size_t fieldBytes = totalCells * sizeof(double);
                    acc_update_self(current, fieldBytes);
#endif

                    if (current == fieldA) {
                        writer->writeFrame(step, fieldStorageA);
                    } else {
                        writer->writeFrame(step, fieldStorageB);
                    }
                    hdf5Time += hdf5Timer.elapsedSeconds();
                }
                ++outputFrames;
                hasLastWrittenStep = true;
                lastWrittenStep = step;
            };

            // As in the serial baseline, this starts after initialization and
            // includes statistics plus CSV/HDF5 finalization.
            ScopedTimer loopTimer;

            if (shouldWriteStep(0, cfg.timeSteps, cfg.outputEvery)) {
                writeOutputFrame(0);
            }

            for (int step = 1; step <= cfg.timeSteps; ++step) {
                ScopedTimer stepTimer;
                advanceTemperatureFieldOpenAcc(
                    current,
                    next,
                    totalCells,
                    cfg.gridWidth,
                    cfg.gridHeight,
                    coeffs
                );
                pureDynamicsTime += stepTimer.elapsedSeconds();
                std::swap(current, next);

                if (shouldWriteStep(
                        step, cfg.timeSteps, cfg.outputEvery
                    )) {
                    writeOutputFrame(step);
                }
            }

            if (writer) writer->close();
            csv.flush();

            loopWallTime = loopTimer.elapsedSeconds();
            totalWallTime = totalTimer.elapsedSeconds();
        }

        const double updates =
            static_cast<double>(cfg.gridWidth - 2)
            * static_cast<double>(cfg.gridHeight - 2)
            * static_cast<double>(cfg.timeSteps);

        std::cout << "Weight field time:             " << weightTime << " s\n";
        std::cout << "Weight range reduction time:   " << weightRangeTime << " s\n";
        std::cout << "Initialization time:           " << initTime << " s\n";
        std::cout << "Pure dynamics compute time:    " << pureDynamicsTime << " s\n";
        std::cout << "Statistics time:               " << statisticsTime << " s\n";
        std::cout << "CSV write time:                " << csvTime << " s\n";
        std::cout << "HDF5 write time:               " << hdf5Time << " s\n";
        std::cout << "Dynamics loop wall time:       " << loopWallTime << " s\n";
        std::cout << "Total measured wall time:      " << totalWallTime << " s\n";
        std::cout << "Output frames:                 " << outputFrames << '\n';

        if (cfg.timeSteps > 0 && pureDynamicsTime > 0.0) {
            std::cout << "Pure dynamics performance:     "
                      << updates / pureDynamicsTime / 1.0e9 << " GLUP/s\n";
        }
        if (cfg.timeSteps > 0 && loopWallTime > 0.0) {
            std::cout << "Loop end-to-end performance:   "
                      << updates / loopWallTime / 1.0e9 << " GLUP/s\n";
        }

        std::cout << "Mean discrepancy:              "
                  << std::setprecision(15) << meanDiscrepancy << '\n';
        std::cout << "Final min:                     " << finalStats.minValue << '\n';
        std::cout << "Final mean:                    " << finalStats.meanValue << '\n';
        std::cout << "Final max:                     " << finalStats.maxValue << '\n';
        std::cout << "Final std.dev.:                " << finalStats.stdDev << '\n';
        std::cout << "Final L2 norm:                 " << finalStats.l2Norm << '\n';
        std::cout << "Final checksum:                " << finalStats.checksum << '\n';
        std::cout << "Weight range:                  "
                  << minWeight << " ... " << maxWeight << '\n';
        std::cout << "\nSimulation completed successfully.\n";
        return 0;

#ifdef USE_HDF5
    } catch (const H5::Exception& error) {
        std::cerr << "HDF5 ERROR: " << error.getDetailMsg() << '\n';
        return 1;
#endif
    } catch (const std::exception& error) {
        std::cerr << "CRITICAL ERROR: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "CRITICAL ERROR: unknown failure\n";
        return 1;
    }
}
