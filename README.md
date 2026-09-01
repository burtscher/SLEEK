
# SLEEK: Compressing Memcpy for Floating-Point Data on GPUs

This repository provides our GPU and CPU implementation of SLEEK, a new main-memory compression algorithm for IEEE 754-formatted single- and double-precision floating-point data that delivers speeds comparable to or exceeding the speed of CUDA _memcpy_ on GPUs. Also included are the relevant scripts to download the SDRBench inputs, compile and run our code, gather results, and generate figures similar to those shown in our IPDPS'26 paper. For more information on SLEEK, see the paper listed in the [Publication](#Publication) section below.

SLEEK provides lossless and lossy algorithms that yield substantial compression ratios at high speeds. They support all floating-point values, including NaNs, infinities, and denormals. The lossy compression algorithm guarantees point-wise absolute error bounds without the need to check for violations. The GPU and CPU implementations of SLEEK produce compressed and decompressed outputs that are bit-for-bit identical, allowing data compressed on a GPU to be decompressed on a CPU and vice versa. SLEEK delivers much higher throughput at a given compression ratio compared to the state-of-the-art algorithms from the literature.

For instructions on how to compile and run our SLEEK codes, see [Using SLEEK](#Using-SLEEK) below. The [Single-Precision](#Single-Precision-float32) section provides a quick description for single-precision inputs, and the [Double-Precision](#Double-Precision-float64) section provides a description for double-precision inputs.


For instructions on the artifact, see [Artifact Description](#Artifact-Description) below.

## Using SLEEK

SLEEK requires a CUDA-capable device for the GPU implementation and an x86 CPU for the CPU implementation. We evaluated SLEEK on 3 different GPUs in our paper, compiling with version 12.0 of the NVIDIA CUDA compiler and version 523.85 of the NVIDIA driver. Older versions may work but are untested. For the CPU codes, a g++ compiler with OpenMP support is required. Links to the software versions used are in the [Artifact Description](#Artifact-Description) below. When using SLEEK with your own inputs, your files must be binary files containing only IEEE 754-formatted floating-point values.

### Quick Start

#### Single Precision (float32)

To use SLEEK with your own single-precision input, execute the following commands to download and compile our codes.

```bash
git clone https://github.com/burtscher/SLEEK/
cd SLEEK
bash compile_standalone_compressors_single.sh
```

See the following commands on how to compress and decompress your input. Replace `input_file_name` with the path to the input you want to compress, and `compressed_file_name`/`decompressed_file_name` with your desired name for the compressed and decompressed file. For lossy compression, replace `error_bound` with your desired error bound (e.g., "0.01" without the quotes).

```bash
# Lossless compression
./sleek_single_compressor_lossless input_file_name compressed_file_name

# Lossless decompression
./sleek_single_decompressor_lossless compressed_file_name decompressed_file_name

# Lossy compression
./sleek_single_compressor_lossy input_file_name compressed_file_name error_bound

# Lossy decompression
./sleek_single_decompressor_lossy compressed_file_name decompressed_file_name error_bound
```

#### Double Precision (float64)

To use SLEEK with your own double-precision input, execute the following commands to download and compile our double-precision codes.
```bash
git clone https://github.com/burtscher/SLEEK/
cd SLEEK
bash compile_standalone_compressors_double.sh
```

See the following commands on how to compress and decompress your input. Replace `input_file_name` with the path to the input you want to compress, and `compressed_file_name`/`decompressed_file_name` with your desired name for the compressed and decompressed file. For lossy compression, replace `error_bound` with your desired error bound (e.g., "0.01" without the quotes).

```bash
# Lossless compression
./sleek_double_compressor_lossless input_file_name compressed_file_name

# Lossless decompression
./sleek_double_decompressor_lossless compressed_file_name decompressed_file_name

# Lossy compression
./sleek_double_compressor_lossy input_file_name compressed_file_name error_bound

# Lossy decompression
./sleek_double_decompressor_lossy compressed_file_name decompressed_file_name error_bound
```

### Detailed Explanation

The codes can be found in 4 different folders, separated by lossless and lossy CPU and GPU implementations. Each folder contains a separate compressor and decompressor `.cu` or `.cpp` file for single- and double-precision, for a total of 4 files.
- [Lossless GPU Codes](./lossless_compressors_gpu)
- [Lossy GPU Codes](./lossy_compressors_gpu)
- [Lossless CPU Codes](./lossless_compressors_cpu)
- [Lossy CPU Codes](./lossy_compressors_cpu)

These codes use our SLEEK API, which can be found in the [include folder](./include). The API provides an easier means of integrating SLEEK into other CPU and GPU codes. For examples of how to use the SLEEK API, please refer to the aforementioned GPU and CPU code folders.

The `compile_compressors_single.sh` and `compile_compressors_double.sh` Bash scripts compile all CPU and GPU codes for single- and double-precision inputs, respectively. These scripts automatically use the appropriate `-arch` flag for the installed GPU.

To compile SLEEK for use with single-precision inputs, use the following command while in the top-level directory of this repository.
```bash
bash compile_compressors_single.sh
```

For an example of how to compile an individual GPU code, see the following command. Replace `sm_XX` with your correct GPU architecture.
```bash
nvcc -O3 -arch=sm_XX ./lossless_compressors_gpu/sleek_single_compressor.cu -o sleek_single_compressor_lossless
```

For an example of how to compile an individual CPU code, see the following command.
```bash
g++ -O3 -fopenmp -march=native -std=c++17 ./lossless_compressors_cpu/sleek_single_compressor.cpp -o sleek_single_compressor_lossless_cpu
```

Please see `compile_compressors_single.sh` and `compile_compressors_double.sh` for more examples of compiling a single code.

## Artifact Description

To run our scripts and generate the figures, the following software needs to be installed. Older versions of these codes may work but are untested.
- Python version 3.13: https://www.python.org/downloads/release/python-3130/
- Matplotlib Python visualization library version 3.10.1: https://pypi.org/project/matplotlib/3.10.1/


We used several datasets from the SDRBench suite. Our scripts automatically download the inputs used in the paper from this URL: https://sdrbench.github.io/. Furthermore, because the total size of the inputs is over 45 gigabytes, we provide an option to download a subset of the inputs we used. To download the subset, set the flag inside of the `full-workflow.sh` script named `use_limited_input_set` to `true`.


To compile our codes, the nvcc compiler and a g++ compiler with OpenMP support are required.
- NVIDIA CUDA compiler version 12.0: https://developer.nvidia.com/cuda-12-0-0-download-archive
- NVIDIA driver version 525.85: https://www.nvidia.com/en-us/drivers/details/198554/
- GNU Compiler Collection version 12.2: https://ftp.gnu.org/gnu/gcc/gcc-12.2.0/


There is a single script named `full-workflow.sh` that completes the following steps.
1) It downloads the single- and double-precision inputs from SDRBench.
2) It compiles and executes the SLEEK GPU codes on those inputs.
3) It verifies that the CPU and GPU codes output the bit-for-bit same compressed and decompressed file for the given inputs.
4) It generates figures showing the compression ratio, encoding throughput, and decoding throughput of the GPU codes.

To run this artifact, execute the following commands.

```bash
git clone https://github.com/burtscher/SLEEK/
cd SLEEK
bash check-deps.sh
bash full-workflow.sh
```

The generated figures will be stored in the directory named `figures`.

## Publication

If you use SLEEK in your work, please cite the following publication:

Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher. "SLEEK: Compressing Memory Copies for Floating-Point Data on GPUs." Proceedings of the 40th IEEE International Parallel and Distributed Processing Symposium. May 2026. [[paper](https://userweb.cs.txstate.edu/~burtscher/papers/ipdps26.pdf)] [[slides](https://userweb.cs.txstate.edu/~burtscher/slides/ipdps26.pptx)] [[doi](https://doi.org/10.1109/IPDPS65963.2026.00022)]

*This material is based upon work supported by the National Science Foundation under Grant #2403380 and by the Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under Award #DE-SC0022223.*
