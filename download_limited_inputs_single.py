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

import requests
import re
from urllib.parse import urljoin
import os
import time
import shutil
import tarfile
import subprocess


def unzip_files(directory="."):
    for filename in os.listdir(directory):
        if filename.endswith(".tar.gz"):
            print("Extracting", filename)
            # Using subprocess to run the tar command
            subprocess.run(["tar", "-xvzf", os.path.join(directory, filename), "-C", directory])
            print("Extracted", filename)

def get_links_from_page(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Use regular expressions to find all links in the HTML content
        links = re.findall(r'href=[\'"]?([^\'" >]+)', response.text)
        
        # Extract the absolute URLs from the found links
        extracted_links = [urljoin(url, link) for link in links]
        
        return extracted_links
    else:
        print("Failed to fetch page. Status code:", response.status_code)
        return []

def download_file(url, directory="."):
    # Extract filename from URL
    filename = url.split("/")[-1]
    
    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(os.path.join(directory, filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def delete_files_with_word(directory, word):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if word in filename:
                filepath = os.path.join(root, filename)
                try:
                       os.remove(filepath)
                       print("Deleted", filepath)
                except Exception as e:
                       print(f"Error deleting {filepath}: {e}")



def delete_gz_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".gz"):
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    print("Deleted", filepath)
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

# Timing the script
start_time = time.time()

# Example usage:
url = "https://sdrbench.github.io/"

# full single-precision dataset
keywords = ["SDRBENCH-exaalt-copper.tar",
            "SDRBENCH-QMCPack.tar"]

# check if files are already downloaded
folder_names = ["exaalt",
                "dataset"]

keywords_tmp = list()
for idx, fname in enumerate(folder_names):
    if not os.path.exists(f"./single_inputs/{fname}"):
        keywords_tmp.append(keywords[idx])

keywords = keywords_tmp

if len(keywords) == 0:
    print("All inputs already downloaded!")
    quit()

links = get_links_from_page(url)
for link in links:
    if link.endswith('.gz'):
        for keyword in keywords:
            if keyword in link:
                print("Downloading", link)
                download_file(link)
                print("Downloaded", link)
                break  # Break the inner loop to avoid downloading the same file multiple times if it matches multiple keywords

# Create a folder named "single_inputs" and move all downloaded files into it
if not os.path.exists("single_inputs"):
    os.makedirs("single_inputs")
for filename in os.listdir("."):
    if filename.endswith(".gz"):
        shutil.move(filename, os.path.join("single_inputs", filename))

# Unzip all .tar.gz files in the "single_inputs" folder
unzip_files("single_inputs")

delete_gz_files("single_inputs")

# Delete files from "single_inputs" folder that contain the word "log" and have the .txt extension
delete_files_with_word("single_inputs", "log")
delete_files_with_word("single_inputs", ".txt")

# Move the dataset files so the next script can see them
os.system("mv single_inputs/dataset/*/* single_inputs/dataset/")


# Define the base directory
base_dir = "single_inputs"

# Move all files from specified folders into the new 'exaalt' folder
folders_to_move = [
    "SDRBENCH-exaalt-copper",
    "SDRBENCH-exaalt-helium",
    "2869440"
    ]
exaalt_folder = os.path.join(base_dir, "exaalt")
os.makedirs(exaalt_folder, exist_ok=True)

for folder in folders_to_move:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                shutil.move(os.path.join(root, file), os.path.join(exaalt_folder, file))

# Delete the three folders
for folder in folders_to_move:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)


# Calculate execution time
execution_time = time.time() - start_time
print(f"Script execution time: {execution_time:.1f} seconds")

    
