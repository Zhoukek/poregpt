#!/usr/bin/env python3
"""
Script to sample FAST5 files from a source directory and copy them to a destination,
preserving the directory structure.
"""

import argparse
import os
import random
from pathlib import Path
import shutil

def get_fast5_files_in_dir(directory: Path) -> list:
    """
    Gets all .fast5 files in a single directory.

    Args:
        directory (Path): The directory to search in.

    Returns:
        list: A list of Path objects for .fast5 files.
    """
    return [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == '.fast5']

def main():
    """
    Main function to parse arguments and orchestrate the sampling process.
    """
    parser = argparse.ArgumentParser(
        description="Recursively sample a percentage of .fast5 files from a source directory "
                    "and copy them to a destination directory, preserving subdirectory structure."
    )
    parser.add_argument(
        'sampling_ratio',
        type=float,
        help='The ratio of files to select (e.g., 0.1 for 10%%, 0.5 for 50%%).'
    )
    parser.add_argument(
        'input_directory',
        type=Path,
        help='The path to the source directory containing .fast5 files.'
    )
    parser.add_argument(
        'output_directory',
        type=Path,
        help='The path to the destination directory where sampled files will be copied.'
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_directory.exists() or not args.input_directory.is_dir():
        print(f"Error: Input directory does not exist or is not a directory: {args.input_directory}")
        exit(1)

    # Validate sampling ratio
    if not (0.0 <= args.sampling_ratio <= 1.0):
        print("Error: Sampling ratio must be between 0.0 and 1.0.")
        exit(1)

    # Ensure output directory exists
    args.output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Starting sampling process...")
    print(f"Source: {args.input_directory.absolute()}")
    print(f"Destination: {args.output_directory.absolute()}")
    print(f"Sampling Ratio: {args.sampling_ratio * 100:.2f}%\n")

    # Walk through the source directory tree
    processed_dirs = 0
    total_selected_files = 0
    for root, dirs, files in os.walk(args.input_directory):
        current_source_dir = Path(root)
        
        # Get all fast5 files in the current directory
        fast5_files_in_current_dir = get_fast5_files_in_dir(current_source_dir)
        
        if not fast5_files_in_current_dir:
            continue  # Skip directories with no .fast5 files

        num_to_select = int(len(fast5_files_in_current_dir) * args.sampling_ratio)
        
        # Randomly sample the required number of files
        selected_files = random.sample(fast5_files_in_current_dir, num_to_select)

        # Calculate the corresponding output directory path
        relative_path = current_source_dir.relative_to(args.input_directory)
        current_output_dir = args.output_directory / relative_path
        current_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy each selected file to the output directory
        for file in selected_files:
            destination_file = current_output_dir / file.name
            shutil.copy2(file, destination_file) # copy2 also copies metadata
            total_selected_files += 1
            
        print(f"Selected {len(selected_files)} out of {len(fast5_files_in_current_dir)} files "
              f"from '{current_source_dir}' -> '{current_output_dir}'")
        
        processed_dirs += 1
        
    print("\nSampling process completed successfully!")
    print(f"Total directories processed: {processed_dirs}")
    print(f"Total files copied: {total_selected_files}")

if __name__ == "__main__":
    main()
