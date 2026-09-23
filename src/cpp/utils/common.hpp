#pragma once

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "types.hpp"

namespace fs = std::filesystem;

int parseStrictInt(const std::string& text, const std::string& what) {
    int value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();

    const auto [ptr, ec] = std::from_chars(begin, end, value);

    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid " + what + ": '" + text + "'");
    }

    return value;
}

bool beginsWithDoubleDash(const std::string& text) {
    return text.rfind("--", 0) == 0;
}

bool isNoHdf5Token(const std::string& text) {
    return text == "none" || text == "NONE" || text == "-" || text == "--no-hdf5";
}

void ensureParentDirectoryExists(const std::string& fileName) {
    if (fileName.empty() || isNoHdf5Token(fileName)) {
        return;
    }

    const fs::path path(fileName);
    const fs::path parent = path.parent_path();

    if (parent.empty()) {
        return;
    }

    std::error_code ec;

    if (fs::exists(parent, ec)) {
        if (!fs::is_directory(parent, ec)) {
            throw std::runtime_error("Output parent exists but is not a directory: " + parent.string());
        }
        return;
    }

    if (!fs::create_directories(parent, ec) && ec) {
        throw std::runtime_error("Cannot create output directory '" + parent.string() + "': " + ec.message());
    }
}

SimulationConfig readConfigurationFile(const std::string& fileName) {
    std::ifstream input(fileName);

    if (!input) {
        throw std::runtime_error("Cannot open input file: " + fileName);
    }

    std::vector<std::string> tokens;
    std::string line;

    while (std::getline(input, line)) {
        const auto commentPos = line.find('#');

        if (commentPos != std::string::npos) {
            line.erase(commentPos);
        }

        std::istringstream iss(line);
        std::string token;

        while (iss >> token) {
            tokens.push_back(token);
        }
    }

    if (tokens.empty()) {
        throw std::runtime_error("Input file contains no tokens: " + fileName);
    }

    std::size_t pos = 0;

    auto nextInt = [&]() -> int {
        if (pos >= tokens.size()) {
            throw std::runtime_error("Malformed input: missing integer token");
        }

        return parseStrictInt(tokens[pos++], "integer token");
    };

    auto nextDouble = [&]() -> double {
        if (pos >= tokens.size()) {
            throw std::runtime_error("Malformed input: missing floating-point token");
        }

        const std::string token = tokens[pos++];

        try {
            std::size_t used = 0;
            const double value = std::stod(token, &used);

            if (used != token.size()) {
                throw std::runtime_error("invalid trailing characters");
            }

            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite value");
            }

            return value;
        } catch (...) {
            throw std::runtime_error("Malformed input: invalid floating-point token '" + token + "'");
        }
    };

    SimulationConfig cfg{};

    const int rawWidth = nextInt();
    const int rawHeight = nextInt();

    if (rawWidth < 3 || rawHeight < 3) {
        throw std::runtime_error("Grid dimensions must be at least 3 x 3");
    }

    cfg.gridWidth = static_cast<std::size_t>(rawWidth);
    cfg.gridHeight = static_cast<std::size_t>(rawHeight);

    const int measuredCount = nextInt();

    if (measuredCount < 0) {
        throw std::runtime_error("Number of measured points cannot be negative");
    }

    cfg.measuredPoints.resize(static_cast<std::size_t>(measuredCount));

    for (int i = 0; i < measuredCount; ++i) {
        auto& p = cfg.measuredPoints[static_cast<std::size_t>(i)];
        p.x = nextDouble();
        p.y = nextDouble();
        p.value = nextDouble();
    }

    cfg.domainStartX = nextDouble();
    cfg.domainStartY = nextDouble();
    cfg.domainWidth = nextDouble();
    cfg.domainHeight = nextDouble();
    cfg.maxFractalIterations = nextInt();
    cfg.timeSteps = nextInt();

    if (cfg.domainWidth <= 0.0 || cfg.domainHeight <= 0.0) {
        throw std::runtime_error("domainWidth and domainHeight must be positive");
    }

    if (cfg.maxFractalIterations <= 0) {
        throw std::runtime_error("maxFractalIterations must be positive");
    }

    if (cfg.timeSteps < 0) {
        throw std::runtime_error("timeSteps must be non-negative");
    }

    if (pos < tokens.size()) {
        cfg.outputEvery = nextInt();

        if (cfg.outputEvery < 0) {
            throw std::runtime_error("outputEvery must be non-negative");
        }
    }

    if (pos != tokens.size()) {
        throw std::runtime_error("Malformed input: unexpected extra tokens at end of file");
    }

    return cfg;
}

GridMapping buildGridMapping(const SimulationConfig& cfg) {
    GridMapping mapping{};
    mapping.x0 = cfg.domainStartX;
    mapping.y0 = cfg.domainStartY;
    mapping.dx = cfg.domainWidth  / static_cast<double>(cfg.gridWidth  - 1);
    mapping.dy = cfg.domainHeight / static_cast<double>(cfg.gridHeight - 1);
    return mapping;
}

double computeMeanDiscrepancy(const SimulationConfig& cfg) {
    if (cfg.measuredPoints.empty()) {
        return 0.0;
    }

    double sum = 0.0;

    for (const auto& p : cfg.measuredPoints) {
        sum += p.value - analyticalReferenceField(p.x, p.y);
    }

    return sum / static_cast<double>(cfg.measuredPoints.size());
}

UpdateCoefficients buildUpdateCoefficients(double dx, double dy, double damping = 100.0) {
    if (dx <= 0.0 || dy <= 0.0) {
        throw std::invalid_argument("Grid spacing must be positive");
    }

    if (damping <= 0.0) {
        throw std::invalid_argument("Damping parameter must be positive");
    }

    UpdateCoefficients c{};
    c.damping = damping;
    c.stepX = dx;
    c.stepY = dy;
    c.laplaceX = -2.0 * (1.0 + c.damping * c.stepX / (c.stepX * c.stepX + c.damping));
    c.laplaceY = -2.0 * (1.0 + c.damping * c.stepY / (c.stepY * c.stepY + c.damping));
    c.coeffX = (c.stepX + c.damping * std::exp(c.stepX)) / (15.0 * c.damping + c.stepX);
    c.coeffY = (c.stepY + c.damping * std::exp(c.stepY)) / (15.0 * c.damping + c.stepY);
    return c;
}

void writeStatisticsHeader(std::ostream& out) {
    out << "Step;Min;Mean;Max;Std_dev;L2_norm;Checksum\n";
}

void writeStatisticsRow(std::ostream& out, int step, const FieldStatistics& stats) {
    out << step << ';'
        << std::setprecision(15) << stats.minValue  << ';'
        << std::setprecision(15) << stats.meanValue << ';'
        << std::setprecision(15) << stats.maxValue  << ';'
        << std::setprecision(15) << stats.stdDev    << ';'
        << std::setprecision(15) << stats.l2Norm    << ';'
        << std::setprecision(15) << stats.checksum  << '\n';
}

CommandLineOptions parseCommandLineArguments(int argc, char** argv) {
    CommandLineOptions options{};
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--no-hdf5") {
            options.writeHdf5 = false;
            options.h5File = "none";
            continue;
        }

        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage:\n"
                << "  " << argv[0] << " [inputFile] [h5File|none|--no-hdf5] [csvFile] [outputEvery]\n\n"
                << "Examples:\n"
                << "  " << argv[0] << " input/Cooling.in none                  output/Cooling_cpp.csv  0\n"
                << "  " << argv[0] << " input/Cooling.in output/Cooling_cpp.h5 output/Cooling_cpp.csv 50\n\n"
                << "Default official grading mode disables HDF5.\n";
            std::exit(0);
        }

        if (beginsWithDoubleDash(arg)) {
            throw std::runtime_error("Unknown option: " + arg);
        }

        positional.push_back(arg);
    }

    if (positional.size() > 4) {
        throw std::runtime_error("Too many positional arguments");
    }

    if (positional.size() >= 1) {
        options.inputFile = positional[0];
    }

    if (positional.size() >= 2) {
        options.h5File = positional[1];
        options.writeHdf5 = !isNoHdf5Token(options.h5File);
    }

    if (positional.size() >= 3) {
        options.csvFile = positional[2];
    }

    if (positional.size() >= 4) {
        options.outputEvery = parseStrictInt(positional[3], "outputEvery");
        options.overrideOutputEvery = true;

        if (options.outputEvery < 0) {
            throw std::runtime_error("outputEvery must be non-negative");
        }
    }

    return options;
}

bool shouldWriteStep(int step, int finalStep, int outputEvery) {
    if (step == finalStep) {
        return true;
    }

    if (outputEvery <= 0) {
        return false;
    }

    if (step == 0) {
        return true;
    }

    return (step % outputEvery) == 0;
}

void printRunHeader(
    const CommandLineOptions& cli,
    const SimulationConfig& cfg,
    bool hdf5Compiled
) {
    std::cout << "Input file:                    " << cli.inputFile << '\n';
    std::cout << "CSV output:                    " << cli.csvFile << '\n';
    std::cout << "HDF5 compiled:                 " << (hdf5Compiled ? "yes" : "no") << '\n';
    std::cout << "HDF5 output:                   " << (cli.writeHdf5 ? cli.h5File : "disabled") << '\n';
    std::cout << "Official grading mode:         " << (!cli.writeHdf5 ? "yes" : "no") << '\n';
    std::cout << "Grid:                          " << cfg.gridWidth << " x " << cfg.gridHeight << '\n';
    std::cout << "Measured points:               " << cfg.measuredPoints.size() << '\n';
    std::cout << "Max fractal iterations:        " << cfg.maxFractalIterations << '\n';
    std::cout << "Time steps:                    " << cfg.timeSteps << '\n';

    if (cfg.outputEvery == 0) {
        std::cout << "Snapshot/statistics policy:     final step only\n";
    } else {
        std::cout << "Snapshot/statistics policy:     step 0, every " << cfg.outputEvery << " step(s), and final step\n";
    }

    std::cout << '\n';
}
