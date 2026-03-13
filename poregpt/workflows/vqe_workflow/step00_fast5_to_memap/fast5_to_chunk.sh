#!/bin/bash

# =============================================================================
# Fast5 to NPY Converter Script
# 
# Description:
#   This script converts Oxford Nanopore Technologies (ONT) fast5 files to 
#   numpy array (.npy) format after applying signal processing and chunking.
#   It uses multiprocessing for efficient batch conversion.
#
# Configuration:
#   Edit the variables below to customize the conversion behavior.
#   The script will use these predefined values without requiring command line arguments.
#
# Requirements:
#   - Python environment with required packages installed
#   - Sufficient disk space for output files
#   - Read permissions for input directory
#   - Write permissions for output directory
# =============================================================================

# Exit immediately if any command fails
set -euo pipefail

# Global variables
SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_TIME=$(date +%s)

# =============================================================================
# USER CONFIGURATION SECTION
# Edit these variables to configure the conversion behavior
# =============================================================================

# Input directory containing fast5 files (will search recursively)
INPUT_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655_10p/fast5"

# Output directory to save npy files
OUTPUT_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655_10p/chunk_apple"

# Simple Fast5 to NPY converter script
# Edit the variables below to configure the conversion

# Configuration
STRATEGY="apple"                           # Signal processing strategy
CHUNK_SIZE=40000                          # Size of each signal chunk
OVERLAP_SIZE=4000                        # Overlap between chunks
PROCESSES=32                               # Number of parallel processes

# Validate directories
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting conversion..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Strategy: $STRATEGY"
echo "Chunk size: $CHUNK_SIZE"
echo "Overlap: $OVERLAP_SIZE"
echo "Processes: $PROCESSES"

# Run the Python script
python3 -u fast5_to_trank.py  \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --strategy "$STRATEGY" \
    --chunk-size "$CHUNK_SIZE" \
    --overlap-size "$OVERLAP_SIZE" \
    --processes "$PROCESSES"

echo "Conversion completed!"
