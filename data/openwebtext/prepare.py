"""
Downloads and prepares the Fineweb10B dataset for training.
This script downloads pre-tokenized GPT-2 tokens from Hugging Face,
saving significant preprocessing time.
"""

import os
import sys
import numpy as np
from huggingface_hub import hf_hub_download

# Default to 8 chunks (800M tokens) for manageable download size
DEFAULT_NUM_CHUNKS = 8

def download_file(fname, local_dir):
    """Download a file from HuggingFace if it doesn't exist locally."""
    filepath = os.path.join(local_dir, fname)
    if not os.path.exists(filepath):
        print(f"Downloading {fname}...")
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2", 
            filename=fname,
            repo_type="dataset", 
            local_dir=local_dir
        )
    else:
        print(f"File {fname} already exists, skipping download.")
    return filepath

def concatenate_chunks(chunk_files, output_file):
    """Concatenate multiple binary chunks into a single file."""
    print(f"Concatenating {len(chunk_files)} chunks into {output_file}...")
    
    # First, determine the total size
    total_size = 0
    for chunk_file in chunk_files:
        m = np.memmap(chunk_file, dtype=np.uint16, mode='r')
        total_size += len(m)
        del m  # Close the memmap
    
    # Create output memmap
    print(f"Total tokens: {total_size:,}")
    arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(total_size,))
    
    # Copy chunks
    idx = 0
    for i, chunk_file in enumerate(chunk_files):
        print(f"Processing chunk {i+1}/{len(chunk_files)}...")
        m = np.memmap(chunk_file, dtype=np.uint16, mode='r')
        arr[idx:idx + len(m)] = m
        idx += len(m)
        del m
    
    arr.flush()
    print(f"Created {output_file} with {total_size:,} tokens")

def main():
    # Parse command line arguments
    num_chunks = DEFAULT_NUM_CHUNKS
    if len(sys.argv) >= 2:
        num_chunks = int(sys.argv[1])
        print(f"Downloading {num_chunks} chunks as specified")
    else:
        print(f"Downloading default {num_chunks} chunks (pass a number as argument to change)")
        print(f"Full dataset has 103 chunks. Each chunk is 100M tokens.")
    
    # Create data directory
    data_dir = os.path.dirname(__file__)
    local_dir = os.path.join(data_dir, 'fineweb10B')
    os.makedirs(local_dir, exist_ok=True)
    
    # Download validation set (always just one file)
    print("\n--- Downloading validation set ---")
    val_file = download_file("fineweb_val_000000.bin", local_dir)
    
    # Download training chunks
    print(f"\n--- Downloading {num_chunks} training chunks ---")
    train_chunk_files = []
    for i in range(1, num_chunks + 1):
        fname = f"fineweb_train_{i:06d}.bin"
        chunk_file = download_file(fname, local_dir)
        train_chunk_files.append(chunk_file)
    
    print("\n--- Download complete! ---")
    
    # Create concatenated files in the data directory
    print("\n--- Creating concatenated binary files ---")
    
    # Validation file - just copy/rename
    val_output = os.path.join(data_dir, 'val.bin')
    if not os.path.exists(val_output):
        print("Creating val.bin...")
        # Just create a symlink or copy
        import shutil
        shutil.copy2(val_file, val_output)
        print(f"Created {val_output}")
    else:
        print("val.bin already exists, skipping.")
    
    # Training file - concatenate all chunks
    train_output = os.path.join(data_dir, 'train.bin')
    if not os.path.exists(train_output):
        concatenate_chunks(train_chunk_files, train_output)
    else:
        print("train.bin already exists, skipping concatenation.")
    
    # Print summary
    print("\n--- Summary ---")
    for split in ['train', 'val']:
        filepath = os.path.join(data_dir, f'{split}.bin')
        if os.path.exists(filepath):
            m = np.memmap(filepath, dtype=np.uint16, mode='r')
            print(f"{split}.bin: {len(m):,} tokens ({len(m)/1e6:.1f}M)")
            del m

if __name__ == '__main__':
    main()