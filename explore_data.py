"""
Enhanced script to explore and debug .bin files created by nanoGPT data preparation.
Includes better error handling and debugging for corrupted data.
"""

import os
import sys
import numpy as np
import pickle
import tiktoken

def load_meta(data_dir):
    """Load the meta.pkl file if it exists (for character-level encoding)."""
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta
    return None

def detect_encoding_type(data, data_dir):
    """Detect if this is character-level or BPE encoding."""
    # Check for meta.pkl (character-level encoding)
    meta = load_meta(data_dir)
    if meta and 'itos' in meta:
        return 'char', meta
    
    # Check token range for BPE
    sample = data[:min(1000, len(data))]
    max_token = np.max(sample)
    
    if max_token < 256:  # Likely character-level
        return 'char', None
    else:  # Likely BPE
        return 'bpe', None

def decode_tokens(tokens, encoding_type, meta=None):
    """Decode tokens based on encoding type."""
    if encoding_type == 'char':
        if meta and 'itos' in meta:
            # Use the character mapping from meta.pkl
            itos = meta['itos']
            try:
                return ''.join([itos[i] for i in tokens])
            except (KeyError, IndexError):
                # Fallback to ASCII
                return ''.join([chr(i) if i < 128 else f'[{i}]' for i in tokens])
        else:
            # Assume ASCII encoding
            return ''.join([chr(i) if i < 128 else f'[{i}]' for i in tokens])
    else:  # BPE
        enc = tiktoken.get_encoding("gpt2")
        
        text_parts = []
        for token in tokens:
            if token == 50256:  # EOT token
                text_parts.append("<|endoftext|>")
            elif 0 <= token < enc.n_vocab:  # Valid GPT-2 token range
                try:
                    text = enc.decode([token])
                    text_parts.append(text)
                except Exception:
                    text_parts.append(f"[Token{token}:Error]")
            else:
                # Token outside valid range
                text_parts.append(f"[Invalid:{token}]")
        
        return ''.join(text_parts)

def explore_bin_file(filepath, num_samples=5, sample_length=100):
    """Explore a .bin file by showing multiple decoded samples."""
    print(f"\n{'='*80}")
    print(f"EXPLORING: {filepath}")
    print('='*80)
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found!")
        return
    
    # Load the data
    data_dir = os.path.dirname(filepath)
    data = np.memmap(filepath, dtype=np.uint16, mode='r')
    
    # Detect encoding type
    encoding_type, meta = detect_encoding_type(data, data_dir)
    
    # Show file info
    file_size = os.path.getsize(filepath)
    print(f"\nFile Info:")
    print(f"  Size: {file_size:,} bytes ({file_size/1e9:.2f} GB)")
    print(f"  Total tokens: {len(data):,}")
    print(f"  Encoding type: {encoding_type}")
    
    # Show token statistics
    sample_size = min(10000, len(data))
    sample_data = data[:sample_size]
    print(f"\nToken Statistics (first {sample_size:,} tokens):")
    print(f"  Min token ID: {np.min(sample_data)}")
    print(f"  Max token ID: {np.max(sample_data)}")
    print(f"  Unique tokens: {len(np.unique(sample_data))}")
    
    if encoding_type == 'bpe':
        # Check for invalid GPT-2 tokens
        gpt2_max = 50256
        invalid_mask = sample_data >= gpt2_max
        invalid_count = np.sum(invalid_mask)
        if invalid_count > 0:
            print(f"  WARNING: {invalid_count} tokens exceed GPT-2 vocab size!")
            invalid_tokens = np.unique(sample_data[invalid_mask])
            print(f"  Invalid token IDs: {invalid_tokens.tolist()}")
    
    # Show samples from different parts of the file
    print(f"\n{num_samples} Sample Excerpts:")
    print("-" * 80)
    
    # Calculate positions for samples (evenly distributed)
    file_length = len(data)
    positions = []
    
    if num_samples == 1:
        positions = [0]
    else:
        # Include start, end, and evenly distributed middle positions
        step = (file_length - sample_length) // (num_samples - 1)
        positions = [i * step for i in range(num_samples)]
        positions[-1] = max(0, file_length - sample_length)  # Ensure last position is at end
    
    for i, pos in enumerate(positions):
        # Get tokens
        end_pos = min(pos + sample_length, file_length)
        tokens = data[pos:end_pos].tolist()
        
        # Decode
        text = decode_tokens(tokens, encoding_type, meta)
        
        # Display
        location_pct = (pos / file_length) * 100
        print(f"\nSample {i+1} (position {pos:,}, {location_pct:.1f}% through file):")
        print(f"Tokens {pos} to {end_pos}:")
        
        # Show first few token IDs
        token_preview = tokens[:20]
        print(f"Token IDs: {token_preview}{'...' if len(tokens) > 20 else ''}")
        
        # Show decoded text
        print(f"Decoded text:")
        print("-" * 40)
        # Limit display length and handle special characters
        display_text = text[:500] if len(text) > 500 else text
        display_text = display_text.replace('\n', '\\n')
        print(display_text)
        if len(text) > 500:
            print("... (truncated)")
        print("-" * 40)
    
    # Memory cleanup
    del data

def debug_bin_file(filepath, num_bytes=1000):
    """Debug a .bin file by examining its raw structure."""
    print(f"\n=== DEBUGGING {filepath} ===")
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found!")
        return
    
    # Check file size
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size:,} bytes ({file_size/1e9:.2f} GB)")
    
    # Read as different data types to diagnose
    print("\n1. Reading as uint16 (expected format):")
    data_u16 = np.memmap(filepath, dtype=np.uint16, mode='r')
    print(f"   Shape: {data_u16.shape}")
    print(f"   First 20 values: {data_u16[:20].tolist()}")
    print(f"   Min: {np.min(data_u16[:1000])}, Max: {np.max(data_u16[:1000])}")
    print(f"   Unique values in first 1000: {len(np.unique(data_u16[:1000]))}")
    
    # Check if values are in GPT-2 range
    gpt2_vocab_size = 50257
    out_of_range = np.sum(data_u16[:10000] >= gpt2_vocab_size)
    print(f"   Tokens outside GPT-2 range in first 10k: {out_of_range} ({out_of_range/100:.1f}%)")
    
    # Try reading as raw bytes
    print("\n2. Reading raw bytes:")
    with open(filepath, 'rb') as f:
        raw_bytes = f.read(min(100, file_size))
    print(f"   First 50 bytes (hex): {raw_bytes[:50].hex()}")
    
    # Check byte order
    print("\n3. Checking byte order:")
    # Read same data with different endianness
    data_le = np.frombuffer(raw_bytes[:20], dtype='<u2')  # Little endian
    data_be = np.frombuffer(raw_bytes[:20], dtype='>u2')  # Big endian
    print(f"   Little endian: {data_le.tolist()}")
    print(f"   Big endian: {data_be.tolist()}")
    
    # Try decoding a sample
    print("\n4. Attempting to decode first 100 tokens:")
    enc = tiktoken.get_encoding("gpt2")
    valid_tokens = []
    invalid_count = 0
    
    for i, token in enumerate(data_u16[:100]):
        if 0 <= token < gpt2_vocab_size:
            valid_tokens.append(token)
        else:
            invalid_count += 1
            if invalid_count <= 5:  # Show first 5 invalid tokens
                print(f"   Invalid token at position {i}: {token}")
    
    if valid_tokens:
        try:
            decoded = enc.decode(valid_tokens[:50])
            print(f"   Decoded valid tokens: {decoded[:200]}...")
        except Exception as e:
            print(f"   Decoding error: {e}")
    
    print(f"\n   Total invalid tokens in first 100: {invalid_count}")
    
    # Check for patterns
    print("\n5. Looking for patterns:")
    # Check if all values are the same
    if len(np.unique(data_u16[:1000])) == 1:
        print("   WARNING: All values in first 1000 are the same!")
    
    # Check for alternating pattern (possible byte order issue)
    alternating = True
    for i in range(0, min(20, len(data_u16)-2), 2):
        if data_u16[i] != data_u16[i+2]:
            alternating = False
            break
    if alternating:
        print("   WARNING: Detected alternating pattern - possible byte order issue")
    
    # Look for sequences of zeros
    zero_runs = []
    in_zero_run = False
    run_start = 0
    
    for i in range(min(10000, len(data_u16))):
        if data_u16[i] == 0:
            if not in_zero_run:
                in_zero_run = True
                run_start = i
        else:
            if in_zero_run:
                in_zero_run = False
                run_length = i - run_start
                if run_length > 5:  # Only report runs longer than 5
                    zero_runs.append((run_start, run_length))
    
    if zero_runs:
        print(f"   Found {len(zero_runs)} sequences of zeros longer than 5:")
        for start, length in zero_runs[:5]:  # Show first 5
            print(f"     Position {start}: {length} zeros")
    
    del data_u16  # Close memmap

def find_clean_content_start(data, encoding_type, max_search=10000):
    """Find the first position with clean content."""
    if encoding_type == 'char':
        return 0  # Character encoding is always clean
    
    # For BPE encoding
    gpt2_max = 50256
    i = 0
    
    while i < min(len(data), max_search):
        # Skip invalid tokens
        if data[i] >= gpt2_max:
            i += 1
            continue
            
        # Skip zeros
        if data[i] == 0:
            while i < len(data) and data[i] == 0:
                i += 1
            continue
            
        # Check if we have a reasonable sequence of valid tokens
        valid_count = 0
        j = i
        while j < min(len(data), i + 100) and 0 < data[j] < gpt2_max:
            valid_count += 1
            j += 1
            
        if valid_count >= 50:  # Found clean content
            return i
            
        i += 1
    
    return 0  # Fallback

def compare_datasets(file1, file2, sample_length=200):
    """Compare two .bin files side by side."""
    print("\n" + "="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    for filepath in [file1, file2]:
        if os.path.exists(filepath):
            print(f"\n### {os.path.basename(filepath)} ###")
            data_dir = os.path.dirname(filepath)
            data = np.memmap(filepath, dtype=np.uint16, mode='r')
            
            # Detect encoding
            encoding_type, meta = detect_encoding_type(data, data_dir)
            print(f"Encoding: {encoding_type}")
            
            # Find clean content start
            start_pos = find_clean_content_start(data, encoding_type)
            if start_pos > 0 and encoding_type == 'bpe':
                print(f"Skipping {start_pos} tokens to find clean content...")
            
            # Show a sample from clean content
            end_pos = min(start_pos + sample_length, len(data))
            tokens = data[start_pos:end_pos].tolist()
            text = decode_tokens(tokens, encoding_type, meta)
            
            print(f"Sample of {sample_length} tokens decoded:")
            print("-" * 40)
            print(text[:500] + "..." if len(text) > 500 else text)
            print("-" * 40)
            
            # Show some statistics
            print(f"Total tokens: {len(data):,}")
            print(f"Unique tokens in first 10k: {len(np.unique(data[:10000]))}")
            print(f"Token range: {np.min(data[:10000])} - {np.max(data[:10000])}")
            
            # For BPE, show GPT-2 compliance
            if encoding_type == 'bpe':
                gpt2_max = 50256
                over_max = np.sum(data[:10000] > gpt2_max)
                print(f"Tokens > GPT-2 max in first 10k: {over_max} ({over_max/100:.1f}%)")
                
                # Show additional sample from middle
                if len(data) > 1000000:
                    print("\nAdditional sample from middle of file:")
                    mid_pos = len(data) // 2
                    mid_start = find_clean_content_start(data[mid_pos:mid_pos+10000], encoding_type) + mid_pos
                    mid_tokens = data[mid_start:mid_start+200].tolist()
                    mid_text = decode_tokens(mid_tokens, encoding_type, meta)
                    print("-" * 40)
                    print(mid_text[:300] + "..." if len(mid_text) > 300 else mid_text)
                    print("-" * 40)
            
            del data  # Close memmap
        else:
            print(f"\n{filepath} not found")

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python explore_data.py <path_to_bin_file> [num_samples] [sample_length]")
        print("   or: python explore_data.py compare <file1.bin> <file2.bin>")
        print("   or: python explore_data.py debug <file.bin>")
        print("\nExamples:")
        print("  python explore_data.py data/shakespeare_char/val.bin")
        print("  python explore_data.py data/openwebtext/val.bin 3 200")
        print("  python explore_data.py compare data/shakespeare_char/val.bin data/openwebtext/val.bin")
        print("  python explore_data.py debug data/openwebtext/val.bin")
        sys.exit(1)
    
    if sys.argv[1] == "compare" and len(sys.argv) >= 4:
        # Compare mode
        compare_datasets(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "debug" and len(sys.argv) >= 3:
        # Debug mode
        debug_bin_file(sys.argv[2])
    else:
        # Explore mode
        filepath = sys.argv[1]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        sample_length = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        
        explore_bin_file(filepath, num_samples, sample_length)

if __name__ == "__main__":
    main()