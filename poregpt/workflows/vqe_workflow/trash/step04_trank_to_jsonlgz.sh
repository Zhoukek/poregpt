#!/bin/bash

# --- Configuration Section ---
# Modify these variables according to your setup

# Input directory containing .npy files
INPUT_DIR="/mnt/nas_syy/default/huada_signal_llm/dataset/dna/human_min0_max2_read96655/trank_apple" # <--- CHANGE THIS

# Output directory for .jsonl.gz files
OUTPUT_DIR="/mnt/nas_syy/default/huada_signal_llm/dataset/dna/human_min0_max2_read96655/jsonlgz" # <--- CHANGE THIS

# Path to your trained VQ tokenizer model checkpoint (.pth file)
MODEL_CHECKPOINT="/mnt/nas_syy/default/poregpt/shared/vqe_models_pass20_init_codebook_cnn_and_finetuned/checkpoints/porepgt_vqe_tokenizer.spoch38000.pth" # <--- CHANGE THIS

# Device: 'cuda' for GPU, 'cpu' for CPU
DEVICE="cuda" # Change to 'cpu' if needed

# Batch size for tokenization (adjust based on your GPU memory)
BATCH_SIZE=8 # Adjust as needed

# Number of parallel processes (optional, defaults to CPU count)
NUM_PROCESSES=8 # Set to desired number, or omit for auto-detection

# Path to the Python script

# --- End of Configuration ---

echo "Starting tokenization process..."
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Checkpoint: $MODEL_CHECKPOINT"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "Processes: $NUM_PROCESSES"
echo "Python Script: $PYTHON_SCRIPT"
echo "----------------------------------------"


# Check if the input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: Input directory '$INPUT_DIR' does not exist!"
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Construct the command
CMD=" poregpt-vqe-tokenize-trank \
     -i \"$INPUT_DIR\" \
     -o \"$OUTPUT_DIR\" \
     --model-ckpt \"$MODEL_CHECKPOINT\" \
     --batch-size $BATCH_SIZE"

# Add processes flag if specified
if [[ -n "$NUM_PROCESSES" && "$NUM_PROCESSES" != "" ]]; then
  CMD="$CMD"
fi

# Execute the command
eval $CMD

# Check the exit status of the Python script
if [[ $? -eq 0 ]]; then
  echo "----------------------------------------"
  echo "✅ Tokenization completed successfully!"
else
  echo "----------------------------------------"
  echo "❌ Tokenization failed! Check the logs above."
  exit 1
fi
