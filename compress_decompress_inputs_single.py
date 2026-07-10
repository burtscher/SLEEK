#!/usr/bin/python3 -u

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
ITERATIONS = 3

SLEEK_C_LOSSLESS = "./sleek_single_compressor_lossless"
SLEEK_D_LOSSLESS = "./sleek_single_decompressor_lossless"
SLEEK_C_LOSSY = "./sleek_single_compressor_lossy"
SLEEK_D_LOSSY = "./sleek_single_decompressor_lossy"
GPU_MEMCPY = "./GPUmemcpy"

def run_lossless_with_input(filepath):
	try:
		result = subprocess.run([SLEEK_C_LOSSLESS, filepath, "TEMPC"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_C_LOSSLESS} for {filepath}: {e}")
		quit()

	compressor_output = output

	try:
		result = subprocess.run([SLEEK_D_LOSSLESS, "TEMPC", "TEMPD"], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_D_LOSSLESS} for {filepath}: {e}")
		quit()

	decompressor_output = output

	return (compressor_output, decompressor_output)

def run_lossy_with_input(filepath, errbnd):
	try:
		result = subprocess.run([SLEEK_C_LOSSY, filepath, "TEMPC", errbnd], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_C_LOSSY} for {filepath}: {e}")
		quit()

	compressor_output = output

	try:
		result = subprocess.run([SLEEK_D_LOSSY, "TEMPC", "TEMPD", errbnd, filepath], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {SLEEK_D_LOSSY} for {filepath}: {e}")
		quit()

	decompressor_output = output

	return (compressor_output, decompressor_output)

def run_memcpy_input(filepath):
	try:
		result = subprocess.run([GPU_MEMCPY, filepath], capture_output=True, text=True, check=True)
		output = result.stdout
	except subprocess.CalledProcessError as e:
		print(f"ERROR running {GPU_MEMCPY} for {filepath}: {e}")
		quit()

	return output


# check arguments
if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <input_directory>")
    sys.exit(1)


# lists of results

ERRBND_LIST = ["0.1",
			   "0.01",
			   "0.001",
			   "0.0001",
			   "0.00001",
			   "0.000001"]

ERRBND_NAME_LIST = ["1e-1",
			   		"1e-2",
			   		"1e-3",
			   		"1e-4",
			   		"1e-5",
			   		"1e-6"]

gm_lossless_ratios = list()
gm_lossless_enc_thps = list()
gm_lossless_dec_thps = list()

gm_lossy_ratios = [list() for _ in ERRBND_LIST]
gm_lossy_enc_thps = [list() for _ in ERRBND_LIST]
gm_lossy_dec_thps = [list() for _ in ERRBND_LIST]

gm_gpu_memcpy_thps = list()

# get list of file names
INPUTS_DIR = sys.argv[1]
input_file_folders = glob.glob(f"{INPUTS_DIR}/*")

for input_folder in input_file_folders:

	input_files = glob.glob(f"{input_folder}/**", recursive=True)
	input_files = [x for x in input_files if not os.path.isdir(x)]

	lossless_ratios = list()
	lossless_enc_thps = list()
	lossless_dec_thps = list()

	lossy_ratios = [list() for _ in ERRBND_LIST]
	lossy_enc_thps = [list() for _ in ERRBND_LIST]
	lossy_dec_thps = [list() for _ in ERRBND_LIST]

	gpu_memcpy_thps = list()

	# for getting median
	e_list = list()
	d_list = list()

	# compress and decompress each file
	for filepath in input_files:
		if SHOW_PER_INPUT_RESULTS:
			print(f"{filepath}")

		# lossless
		e_list.clear()
		d_list.clear()
		for _ in range(ITERATIONS):

			comp_o, decomp_o = run_lossless_with_input(filepath)
			comp_o = comp_o.splitlines()
			decomp_o = decomp_o.splitlines()

			ratio = float(comp_o[-3].split()[-1][0:-1])

			enc_thp = float(comp_o[-1].split()[-2])
			e_list.append(enc_thp)

			dec_thp = float(decomp_o[-1].split()[-2])
			d_list.append(dec_thp)

			if SHOW_PER_INPUT_RESULTS:
				print("Lossless")
				print(comp_o[-3])
				print(comp_o[-1])
				print(decomp_o[-1])
				print()

		lossless_ratios.append(ratio)
		lossless_enc_thps.append(statistics.median(e_list))
		lossless_dec_thps.append(statistics.median(d_list))

		# lossy
		for idx, eb in enumerate(ERRBND_LIST):

			e_list.clear()
			d_list.clear()
			for _ in range(ITERATIONS):
				comp_o, decomp_o = run_lossy_with_input(filepath, eb)
				comp_o = comp_o.splitlines()
				decomp_o = decomp_o.splitlines()

				ratio = float(comp_o[-3].split()[-1][0:-1])

				enc_thp = float(comp_o[-1].split()[-2])
				e_list.append(enc_thp)

				dec_thp = float(decomp_o[-1].split()[-2])
				d_list.append(dec_thp)

				if SHOW_PER_INPUT_RESULTS:
					print(f"Lossy - error bound {eb}")
					print(comp_o[-3])
					print(comp_o[-1])
					print(decomp_o[-1])
					print()

			lossy_ratios[idx].append(ratio)
			lossy_enc_thps[idx].append(statistics.median(e_list))
			lossy_dec_thps[idx].append(statistics.median(d_list))

		# gpu memcpy
		e_list.clear()
		d_list.clear()

		for _ in range(ITERATIONS):
			memcpy_o = run_memcpy_input(filepath)
			memcpy_o = memcpy_o.splitlines()

			thp = float(memcpy_o[7].split()[-2])
			e_list.append(thp)

			if SHOW_PER_INPUT_RESULTS:
				print("GPU D2D Memcpy")
				print(memcpy_o[7])
				print()

		gpu_memcpy_thps.append(statistics.median(e_list))

		if SHOW_PER_INPUT_RESULTS:
			print()

	# geomean of this dataset

	lossless_ratio = statistics.geometric_mean(lossless_ratios)
	gm_lossless_ratios.append(lossless_ratio)

	lossless_enc_thp = statistics.geometric_mean(lossless_enc_thps)
	gm_lossless_enc_thps.append(lossless_enc_thp)

	lossless_dec_thp = statistics.geometric_mean(lossless_dec_thps)
	gm_lossless_dec_thps.append(lossless_dec_thp)

	print(f"Geomean metrics of {input_folder} directory")
	print("* Lossless *")
	print(f"Geometric mean compression ratio: {lossless_ratio:.3f}")
	print(f"Geometric mean encoding throughput: {lossless_enc_thp:.3f} Gbytes/s")
	print(f"Geometric mean decoding throughput: {lossless_dec_thp:.3f} Gbytes/s")
	print()

	for idx, eb in enumerate(ERRBND_LIST):

		lossy_ratio = statistics.geometric_mean(lossy_ratios[idx])
		gm_lossy_ratios[idx].append(lossy_ratio)

		lossy_enc_thp = statistics.geometric_mean(lossy_enc_thps[idx])
		gm_lossy_enc_thps[idx].append(lossy_enc_thp)

		lossy_dec_thp = statistics.geometric_mean(lossy_dec_thps[idx])
		gm_lossy_dec_thps[idx].append(lossy_dec_thp)

		print(f"* Lossy - error bound {eb} *")
		print(f"Geometric mean compression ratio: {lossy_ratio:.3f}")
		print(f"Geometric mean encoding throughput: {lossy_enc_thp:.3f} Gbytes/s")
		print(f"Geometric mean decoding throughput: {lossy_dec_thp:.3f} Gbytes/s")
		print()

	gpu_memcpy_thp = statistics.geometric_mean(gpu_memcpy_thps)
	gm_gpu_memcpy_thps.append(gpu_memcpy_thp)

	print("* GPU D2D Memcpy *")
	print(f"Geometric mean throughput: {gpu_memcpy_thp:.3f} Gbytes/s")
	print()
	print()


#
# geometric mean
#

lossless_ratio = statistics.geometric_mean(gm_lossless_ratios)
lossless_enc_thp = statistics.geometric_mean(gm_lossless_enc_thps)
lossless_dec_thp = statistics.geometric_mean(gm_lossless_dec_thps)

print("*** Lossless ***")
print(f"Geometric mean compression ratio: {lossless_ratio:.3f}")
print(f"Geometric mean encoding throughput: {lossless_enc_thp:.3f} Gbytes/s")
print(f"Geometric mean decoding throughput: {lossless_dec_thp:.3f} Gbytes/s")
print()

for idx, eb in enumerate(ERRBND_LIST):

	lossy_ratio = statistics.geometric_mean(gm_lossy_ratios[idx])
	lossy_enc_thp = statistics.geometric_mean(gm_lossy_enc_thps[idx])
	lossy_dec_thp = statistics.geometric_mean(gm_lossy_dec_thps[idx])

	print(f"*** Lossy - error bound {eb} ***")
	print(f"Geometric mean compression ratio: {lossy_ratio:.3f}")
	print(f"Geometric mean encoding throughput: {lossy_enc_thp:.3f} Gbytes/s")
	print(f"Geometric mean decoding throughput: {lossy_dec_thp:.3f} Gbytes/s")
	print()

gpu_memcpy_thp = statistics.geometric_mean(gm_gpu_memcpy_thps)

print("*** GPU D2D Memcpy ***")
print(f"Geometric mean throughput: {gpu_memcpy_thp:.3f} Gbytes/s")

inputs_dir_name = os.path.basename(INPUTS_DIR)
with open(f"{inputs_dir_name}.results", "w") as fout:
	fout.write(f"Compression Ratio\n")

	for idx, eb in enumerate(ERRBND_LIST):
		fout.write(f"{ERRBND_NAME_LIST[idx]},")

	fout.write(f"Lossless\n")

	for idx, eb in enumerate(ERRBND_LIST):
		fout.write(f"{statistics.geometric_mean(gm_lossy_ratios[idx]):.3f},")

	fout.write(f"{lossless_ratio:.3f}\n\n")

	#

	fout.write(f"Compression Throughput\n")

	for idx, eb in enumerate(ERRBND_LIST):
		fout.write(f"{ERRBND_NAME_LIST[idx]},")

	fout.write(f"Lossless,D2D\n")

	for idx, eb in enumerate(ERRBND_LIST):
		fout.write(f"{statistics.geometric_mean(gm_lossy_enc_thps[idx]):.3f},")

	fout.write(f"{lossless_enc_thp:.3f},{gpu_memcpy_thp:.3f}\n\n")

	#

	fout.write(f"Decompression Throughput\n")

	for idx, eb in enumerate(ERRBND_LIST):
		fout.write(f"{ERRBND_NAME_LIST[idx]},")

	fout.write(f"Lossless,D2D\n")
	
	for idx, eb in enumerate(ERRBND_LIST):
		fout.write(f"{statistics.geometric_mean(gm_lossy_dec_thps[idx]):.3f},")

	fout.write(f"{lossless_dec_thp:.3f},{gpu_memcpy_thp:.3f}\n\n")

