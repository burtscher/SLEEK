#!/usr/bin/python33 -u

# This file is part of SLEEK, a set of ultra-fast lossless and guaranteed-error-bounded lossy main-memory compression algorithms for floating-point data on GPUs.
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# URL: The latest version of this code is available at https://github.com/burtscher/SLEEK/.
#
# Publication: This work is described in detail in the following paper.
# Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher. "SLEEK: Compressing Memory Copies for Floating-Point Data on GPUs." Proceedings of the 40th IEEE International Parallel and Distributed Processing Symposium (IPDPS'26). May 2026.
#
# Sponsor: This material is based upon work supported by the U.S. National Science Foundation under Grant Number 2403380 and by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under Award Number DE-SC0022223.

import os
import subprocess
import statistics
import math
import sys
import glob

SHOW_PER_INPUT_RESULTS = False

SLEEK_C_LOSSLESS = "./sleek_single_compressor_lossless"
SLEEK_D_LOSSLESS = "./sleek_single_decompressor_lossless"
SLEEK_C_LOSSY = "./sleek_single_compressor_lossy"
SLEEK_D_LOSSY = "./sleek_single_decompressor_lossy"

SLEEK_C_LOSSLESS_CPU = "./sleek_single_compressor_lossless_cpu"
SLEEK_D_LOSSLESS_CPU = "./sleek_single_decompressor_lossless_cpu"
SLEEK_C_LOSSY_CPU = "./sleek_single_compressor_lossy_cpu"
SLEEK_D_LOSSY_CPU = "./sleek_single_decompressor_lossy_cpu"

def run_lossless_with_input(filepath):
	try:
		result = subprocess.run([SLEEK_C_LOSSLESS, filepath, "TEMPC", "y"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_C_LOSSLESS} for {filepath}: {e}")
		quit()

	try:
		result = subprocess.run([SLEEK_D_LOSSLESS, "TEMPC", "TEMPD", "y"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_D_LOSSLESS} for {filepath}: {e}")
		quit()

	try:
		result = subprocess.run([SLEEK_C_LOSSLESS_CPU, filepath, "TEMPC_cpu", "y"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_C_LOSSLESS_CPU} for {filepath}: {e}")
		quit()

	try:
		result = subprocess.run([SLEEK_D_LOSSLESS_CPU, "TEMPC_cpu", "TEMPD_cpu", "y"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_D_LOSSLESS_CPU} for {filepath}: {e}")
		quit()

	# verify
	try:
		result = subprocess.run(["diff", "TEMPC", "TEMPC_cpu"], capture_output=True, text=True, check=True)
		output1 = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR compressed files to not match with errbnd {errbnd}")
		return False

	try:
		result = subprocess.run(["diff", "TEMPD", "TEMPD_cpu"], capture_output=True, text=True, check=True)
		output2 = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR decompressed files to not match with errbnd {errbnd}")
		return False

	return len(output1) == 0 and len(output2) == 0


def run_lossy_with_input(filepath, errbnd):
	try:
		result = subprocess.run([SLEEK_C_LOSSY, filepath, "TEMPC", errbnd, "y"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_C_LOSSY} for {filepath}: {e}")
		quit()

	try:
		result = subprocess.run([SLEEK_D_LOSSY, "TEMPC", "TEMPD", filepath, errbnd, "y"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_D_LOSSY} for {filepath}: {e}")
		quit()

	try:
		result = subprocess.run([SLEEK_C_LOSSY_CPU, filepath, "TEMPC_cpu", errbnd, "y"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_C_LOSSY_CPU} for {filepath}: {e}")
		quit()

	try:
		result = subprocess.run([SLEEK_D_LOSSY_CPU, "TEMPC_cpu", "TEMPD_cpu", filepath, errbnd, "y"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_D_LOSSY_CPU} for {filepath}: {e}")
		quit()

	# verify
	try:
		result = subprocess.run(["diff", "TEMPC", "TEMPC_cpu"], capture_output=True, text=True, check=True)
		output1 = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR compressed files to not match with errbnd {errbnd}")
		return False

	try:
		result = subprocess.run(["diff", "TEMPD", "TEMPD_cpu"], capture_output=True, text=True, check=True)
		output2 = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR decompressed files to not match with errbnd {errbnd}")
		return False

	return len(output1) == 0 and len(output2) == 0


# check arguments
if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <input_directory>")
    sys.exit(1)

# get list of file names
INPUTS_DIR = sys.argv[1]
input_files = glob.glob(f"{INPUTS_DIR}/**", recursive=True)
input_files = [x for x in input_files if not os.path.isdir(x)]

# counter

ERRBND_LIST = ["0.1",
			   "0.01",
			   "0.001",
			   "0.0001",
			   "0.00001",
			   "0.000001"]

verified_counter = 0

# compress and decompress each file
for filepath in input_files:
	if SHOW_PER_INPUT_RESULTS:
		print(f"{filepath}")

	# lossless
	is_same = run_lossless_with_input(filepath)
	if is_same: verified_counter += 1

	# lossy
	for idx, eb in enumerate(ERRBND_LIST):
		is_same = run_lossy_with_input(filepath, eb)
		if is_same: verified_counter += 1

	if SHOW_PER_INPUT_RESULTS:
		print("CPU/GPU compressed and decompressed outputs match")
		print()

#
# total
#

print(f"Bit-for-bit verified runs over total runs: {verified_counter} / {len(input_files) * 7}")
