# Cooling Field Solver

This repository contains serial baseline implementations of a 2D cooling / field-evolution solver for a parallel programming project.

The main objective is to optimize and parallelize the computationally intensive parts of the application, especially:

- the irregular fractal-weight computation,
- the field initialization,
- the 2D iterative stencil update,
- the global statistical reductions.

Students may implement optimized versions using one or more of the following programming models:

- OpenMP
- MPI
- CUDA
- OpenACC
- Hybrid approaches
- Numba CPU
- Numba CUDA
- CuPy or other GPU-accelerated Python tools, where appropriate

Both C++ and Python baseline implementations are provided.

---

## Repository Structure

```text
.
|-- CMakeLists.txt
|-- CMakePresets.json
|-- input
|   `-- Cooling.in
|-- logs
|-- output
|-- README.md
|-- requirements.txt
|-- scripts
|   |-- env.leonardo.sh
|   |-- env.macos.sh
|   `-- validate_all.sh
|-- src
|   |-- cpp
|   |   `-- cooling.cpp
|   `-- python
|       `-- cooling.py
|-- submit_cpp.sh
|-- submit_cuda.sh
|-- submit_install_pyenv.sh
|-- submit_numba_cuda.sh
|-- submit_numba.sh
|-- submit_omp.sh
|-- submit_python.sh
|-- submit_validate.sh
`-- tools
    |-- validate_cooling_h5.py
    `-- visualize_cooling_h5.py
```

---

## Baseline Implementations

### C++ Baseline

The serial C++17 implementation is located in:

```text
src/cpp/cooling.cpp
```

This version is intended to be optimized and parallelized using, for example:

- OpenMP
- CUDA
- OpenACC
- MPI
- Hybrid CPU/GPU approaches

The C++ code contains the following main computational phases:

1. computation of a Mandelbrot-like fractal weight field,
2. reduction over the weight field,
3. initialization of the temperature field,
4. iterative 2D stencil updates,
5. computation of global statistics,
6. optional CSV and HDF5 output.

The main repeated compute kernel is the 2D stencil update:

```cpp
advanceTemperatureField(...)
```

which internally calls:

```cpp
updateInterior(...)
applyBoundaryConditions(...)
```

The fractal-weight computation:

```cpp
computeFractalWeights(...)
```

is also a relevant target for parallelization.

---

### Python Baseline

A serial Python/NumPy implementation is available in:

```text
src/python/cooling.py
```

This version is intended for students who prefer to work with Python-based acceleration tools such as:

- Numba CPU
- Numba CUDA
- CuPy

The Python code follows the same numerical model and output format as the C++ baseline.

---

## Input File

An example input file is provided in:

```text
input/Cooling.in
```

The input file contains the following numeric tokens, after removing comments beginning with `#`:

```text
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
```

In the provided example, the grid size is:

```text
4000 x 4000
```

with:

```text
maxFractalIterations = 4000
timeSteps            = 200
outputEvery          = 20
```

You are free to modify the input parameters during development and benchmarking.

In particular, you are encouraged to study how the parallel speed-up changes as the grid size, number of time steps, and number of fractal iterations increase.

---

## Output Policy

The solver can produce two types of output:

1. a CSV file containing validation/statistical quantities,
2. an optional HDF5 file containing field snapshots.

In the provided baseline implementations, the CSV file is always produced.

HDF5 output is optional and should normally be disabled for performance benchmarking.

The output frequency is controlled by:

```text
outputEvery
```

with the following meaning:

```text
outputEvery = 0
```

means final-step statistics/output only.

```text
outputEvery > 0
```

means output at step 0, every `outputEvery` steps, and the final step.

---

## Benchmark / No-HDF5 Mode

For performance benchmarking, HDF5 output should normally be disabled.

The recommended C++ benchmark command is:

```bash
./path/to/cooling_serial input/Cooling.in none output/Cooling_cpp.csv 0
```

The recommended Python benchmark command is:

```bash
python3 ./path/to/cooling.py input/Cooling.in none output/Cooling_python.csv 0
```

The argument:

```text
none
```

disables HDF5 output.

The final command-line argument:

```text
0
```

requests final-step statistics only.

This mode avoids the cost of writing large field snapshots to disk.

Please clearly state in your report whether benchmark timings were collected with or without HDF5 output enabled.

---

## Correctness and Validation

To check correctness, compare the final statistics printed by the optimized version against those printed by the original serial implementation.

The program reports quantities such as:

- final minimum value,
- final mean value,
- final maximum value,
- final standard deviation,
- final L2 norm,
- final checksum,
- mean discrepancy,
- weight range.

The CSV output also contains the time evolution of these quantities according to the chosen output schedule.

Small numerical differences are expected and acceptable because of:

- different floating-point reduction orders,
- different thread scheduling,
- fused multiply-add instructions,
- compiler optimizations,
- CPU versus GPU arithmetic differences,
- different parallel reduction implementations.

However, the optimized implementation should remain numerically consistent with the serial baseline.

For HDF5-based validation, use the tools provided in:

```text
tools/
```

For example:

```bash
python3 tools/validate_cooling_h5.py output/reference.h5 output/optimized.h5
```

A convenience validation script is also provided:

```bash
scripts/validate_all.sh
```

---

## CSV Output

The solver writes a CSV file with the following columns:

```text
Step;Min;Mean;Max;Std_dev;L2_norm;Checksum
```

For example, the C++ baseline writes by default to:

```text
output/Cooling_cpp.csv
```

and the Python baseline writes by default to:

```text
output/Cooling_python.csv
```

The checksum is not a physical quantity. It is intended only as a deterministic validation aid to help detect numerical differences between implementations.

---

## HDF5 Output and Visualization

The program can optionally generate `.h5` files containing:

- field snapshots,
- corresponding step numbers.

These files may become large and are mainly intended for debugging or visualization.

To generate an HDF5 output file with the C++ version:

```bash
./path/to/cooling_serial input/Cooling.in output/Cooling_cpp.h5 output/Cooling_cpp.csv 50
```

To generate an HDF5 output file with the Python version:

```bash
python3 src/python/cooling.py input/Cooling.in output/Cooling_python.h5 output/Cooling_python.csv 50
```

The generated HDF5 file can be visualized using:

```bash
python3 tools/visualize_cooling_h5.py output/Cooling_cpp.h5
```

For official performance measurements, it is recommended to disable HDF5 output.

---

## Python Environment Installation

To install the Python environment on Leonardo, see:

```text
submit_install_pyenv.sh
```

The installation procedure is:

```bash
module purge
module load cuda/12.2
module load gcc/12.2.0
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7

python3 -m venv cooling_venv --system-site-packages
source cooling_venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --no-cache-dir -r requirements.txt

deactivate
```

Depending on the exact content of `submit_install_pyenv.sh`, the virtual environment name may differ.

For installation on a local machine, make sure that:

- Python is available,
- CUDA drivers are installed, if CUDA or Numba CUDA will be used,
- the required Python packages in `requirements.txt` are installed,
- `h5py` is installed if HDF5 output is required.

---

## CMake Compilation

Example CMake configuration files are provided:

```text
CMakeLists.txt
CMakePresets.json
```

These have been tested for:

- OpenMP and CUDA on Leonardo,
- OpenMP on macOS Apple Silicon.

The tested compiler toolchains include:

- GNU,
- NVCC,
- Clang.

Other C++17-compatible compilers may also work, provided that they support the required OpenMP/CUDA/HDF5 configuration.

In general, you are free to modify any files, including the compilation CMake files.

---

### Leonardo

Before compiling on Leonardo, source the provided environment script:

```bash
source scripts/env.leonardo.sh
```

Then configure, build, and install with:

```bash
cmake --preset leonardo-a100
cmake --build --preset leonardo-a100 -j
cmake --install build/leonardo-a100
```

---

### macOS Apple Silicon

Before compiling on macOS Apple Silicon, source the provided environment script:

```bash
source scripts/env.macos.sh
```

Then configure, build, and install with:

```bash
cmake --preset macos-arm64
cmake --build --preset macos-arm64 -j
cmake --install build/macos-arm64
```

---

### Generic x86 CPU Without NVIDIA GPU

This configuration has not been extensively tested:

```bash
cmake --preset generic-x86-nogpu
cmake --build --preset generic-x86-nogpu -j
cmake --install build/generic-x86-nogpu
```

---

### Generic x86 CPU With NVIDIA GPU

This configuration has not been extensively tested:

```bash
cmake --preset generic-x86-nvidia
cmake --build --preset generic-x86-nvidia -j
cmake --install build/generic-x86-nvidia
```

---

## Manual Compilation

The C++ code can also be compiled manually.

Without HDF5 support:

```bash
cd src/cpp
g++ -O3 -std=c++17 -Wall -Wextra -pedantic cooling.cpp -o cooling_serial
```

With HDF5 support:

```bash
cd src/cpp
g++ -O3 -std=c++17 -Wall -Wextra -pedantic -DUSE_HDF5 cooling.cpp -o cooling_serial -lhdf5_cpp -lhdf5
```

Depending on the system, the HDF5 compiler wrapper may also be used:

```bash
cd src/cpp
h5c++ -O3 -std=c++17 -Wall -Wextra -pedantic -DUSE_HDF5 cooling.cpp -o cooling_serial
```

On some systems, explicit include and library paths may be required.

---

## Running the Code

### C++ Serial Baseline

From the repository root, after compiling:

```bash
./src/cpp/cooling_serial input/Cooling.in none output/Cooling_cpp.csv 0
```

or, if the executable has been installed elsewhere:

```bash
./path/to/cooling_serial input/Cooling.in none output/Cooling_cpp.csv 0
```

---

### Python Baseline

From the repository root:

```bash
python3 src/python/cooling.py input/Cooling.in none output/Cooling_python.csv 0
```

You can also explicitly disable HDF5 with:

```bash
python3 src/python/cooling.py --no-hdf5 input/Cooling.in none output/Cooling_python.csv 0
```

---

## Command-Line Arguments

### C++ Version

```bash
./cooling_serial [inputFile] [h5File|none|--no-hdf5] [csvFile] [outputEvery]
```

Examples:

```bash
./cooling_serial input/Cooling.in
./cooling_serial input/Cooling.in none output/Cooling_cpp.csv 0
./cooling_serial input/Cooling.in output/Cooling_cpp.h5 output/Cooling_cpp.csv 50
```

---

### Python Version

```bash
python3 cooling.py [options] [inputFile] [h5File|none] [csvFile] [outputEvery]
```

Examples:

```bash
python3 cooling.py input/Cooling.in none output/Cooling_python.csv 0
python3 cooling.py input/Cooling.in output/Cooling_python.h5 output/Cooling_python.csv 50
python3 cooling.py --no-hdf5 input/Cooling.in none output/Cooling_python.csv 0
```

Additional Python options include:

```bash
--h5-tile-y
--h5-tile-x
```

which control the HDF5 chunk tile size.

---

## Job Submission Scripts

Several example submission scripts are provided:

```text
submit_cpp.sh
submit_omp.sh
submit_cuda.sh
submit_python.sh
submit_numba.sh
submit_numba_cuda.sh
submit_validate.sh
submit_install_pyenv.sh
```

These scripts are mainly intended for Leonardo, but can be adapted to other Linux-based HPC systems with minor changes.

---

## Development Guidelines

When implementing an optimized version:

1. Preserve the numerical model.
2. Preserve the input and output formats.
3. Keep the stencil update formula unchanged.
4. Keep the boundary condition behavior unchanged.
5. Compare final statistics against the serial baseline.
6. Benchmark with HDF5 output disabled unless explicitly studying I/O performance.
7. Report the grid size, number of time steps, hardware, compiler, and compilation flags.
8. Report whether CSV and/or HDF5 output were enabled during timing.
9. Discuss strong and/or weak scaling where appropriate.

The main optimization targets are:

- `computeFractalWeights`,
- `initializeTemperatureField`,
- `advanceTemperatureField`,
- `updateInterior`,
- `computeFieldStatistics`.

---

## Notes on Floating-Point Reproducibility

Parallel implementations may not produce bitwise-identical results to the serial baseline.

This is normal.

Differences may arise from:

- different summation orders,
- different reduction trees,
- thread scheduling,
- GPU execution order,
- vectorization,
- fused multiply-add instructions,
- compiler optimization flags.

The optimized implementation should nevertheless remain close to the serial result within a reasonable numerical tolerance.

---

## Performance Metrics

The program reports timing information for several phases, including:

- weight field computation,
- weight range reduction,
- initialization,
- pure dynamics compute time,
- statistics time,
- CSV write time,
- HDF5 write time,
- dynamics loop wall time,
- total measured wall time.

For stencil performance, the code reports:

```text
GLUP/s
```

where `GLUP/s` means billions of lattice updates per second.

The number of lattice updates is computed from the interior grid cells:

```text
(gridWidth - 2) * (gridHeight - 2) * timeSteps
```

---

## Further Instructions

Please read the comments in the source code for additional implementation details and suggestions.

If you find issues in the code or in the provided scripts, please report them to `m.celoria@cineca.it`
