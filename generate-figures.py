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

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys

OUTPUT_DIR = "./figures"

SHOW_DO_NOT_SAVE = False

if len(sys.argv) != 2:
	print(f"Usage: python3 {sys.argv[0]} <name.results>")
	sys.exit(1)

RESULTS_FILE = sys.argv[1]
with open(RESULTS_FILE, "r") as fin:
	data = fin.read()

data = data.strip()
data = data.split("\n\n")

FIG_TITLE = "Single-Precision" if "single" in RESULTS_FILE else "Double-Precision"
COLORS = ["#8e00f4", "#e504d2", "#ed092e", "#fc5a09", "#fcc900", "#0006ed", "#2ed500", "#cf3a99"]
HATCH_7 = ["/", "/", "/", "/", "/", "/", ""]
HATCH_8 = ["/", "/", "/", "/", "/", "/", "", "x"]

for d in data:

	d = d.splitlines()

	y_axis_title = d[0]
	labels = d[1].split(",")
	results = d[2].split(",")
	results = [float(r) for r in results]

	fig, ax = plt.subplots()

	hatches = HATCH_7 if len(results) == 7 else HATCH_8

	ax.bar(labels, results, color=COLORS, edgecolor="black", linewidth=1.5, hatch=hatches)

	if len(results) == 7:
		ax.legend(loc="upper right", ncol=3, handles=[
			mpatches.Patch(edgecolor='black', facecolor="white", label="Lossy", hatch="///"),
			mpatches.Patch(edgecolor='black', facecolor="white", label="Lossless", hatch=""),
			])
	else:
		ax.legend(loc="upper right", ncol=3, handles=[
			mpatches.Patch(edgecolor='black', facecolor="white", label="Lossy", hatch="///"),
			mpatches.Patch(edgecolor='black', facecolor="white", label="Lossless", hatch=""),
			mpatches.Patch(edgecolor='black', facecolor="white", label="CUDA Memcpy", hatch="xxx"),
			])

	full_fig_title = f"{FIG_TITLE} {y_axis_title}s"
	plt.title(full_fig_title)
	ax.set_xlabel("Code")
	ax.set_ylabel(y_axis_title + " (GB/s)" if "Throughput" in y_axis_title else y_axis_title)

	ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
	ax.set_axisbelow(True)

	plt.tight_layout()

	if SHOW_DO_NOT_SAVE:
		plt.show()
	else:
		plt.savefig(f"{OUTPUT_DIR}/{full_fig_title.replace(' ', '.')}.pdf")

print(f"Generated figures can be found in directory {OUTPUT_DIR}")
