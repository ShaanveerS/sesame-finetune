"""
Script to pre-tokenize training/validation data for Sesame finetuning from Hugging Face Hub
and save incrementally in HDF5.

Usage:
python pretokenize_from_hub.py --train_repo_id your_username/dataset_name --val_repo_id your_username/dataset_name --output /path/to/output/data.hdf5
"""

import argparse
from pathlib import Path
# import sqlite3 # No longer needed for metadata loading
# import pandas as pd # No longer needed for metadata loading
import torch
import torchaudio
import h5py
import numpy as np
from tqdm import tqdm
from datasets import load_dataset # Import this

# --- Ensure these are available ---
# Option 1: Define them here if utils.py is not available on H100
# MIMI_SAMPLE_RATE = 24000  # Example, replace with your actual sample rate
# AUDIO_NUM_CODEBOOKS = 8 # Example, replace with your actual number of codebooks

# Option 2: Assume utils.py is in the same directory or PYTHONPATH
try:
    from utils import load_tokenizers, MIMI_SAMPLE_RATE, AUDIO_NUM_CODEBOOKS
except ImportError:
    print("Warning: Could not import from utils.py. Make sure MIMI_SAMPLE_RATE and AUDIO_NUM_CODEBOOKS are defined.")
    # Define fallbacks or raise an error if critical
    MIMI_SAMPLE_RATE = 24000 # Define a default if import fails
    AUDIO_NUM_CODEBOOKS = 8  # Define a default if import fails
    # You'll also need a load_tokenizers function. For this example, we'll assume it's handled.
    # If not, you'd need to define/import it here.
    def load_tokenizers(device):
        print("ERROR: `load_tokenizers` function is missing. Please define or import it.")
        raise NotImplementedError("`load_tokenizers` must be available.")
# --- End Ensure ---

def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    # Changed to accept repo IDs
    parser.add_argument("--train_repo_id", type=str, required=True, help="Hugging Face Hub repo ID for training data (e.g., username/dataset_name).")
    parser.add_argument("--val_repo_id", type=str, required=True, help="Hugging Face Hub repo ID for validation data (e.g., username/dataset_name, can be same as train).")
    # Potentially specify different splits if train/val are in the same repo_id but different configurations or splits
    parser.add_argument("--train_split_name", type=str, default="train", help="Split name for training data (default: train).")
    parser.add_argument("--val_split_name", type=str, default="validation", help="Split name for validation data (default: validation).")

    parser.add_argument("--output", type=Path, default="./data/tokens.hdf5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_every", type=int, default=100, help="Save every N samples")
    parser.add_argument("--omit_speaker_id", action="store_true", help="Don't prepend text with a speaker id")
    args = parser.parse_args(arg_string.split() if arg_string else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args

# load_metadata function is no longer needed as we use datasets.load_dataset

def append_to_hdf5(file_path, split, audio_tokens_batch, text_tokens_batch, compression="gzip"):
    """
    Append audio, text, and length information to the HDF5 file.
    Audio is flattened (vlen) for space efficiency.
    """
    with h5py.File(file_path, "a") as f:
        grp = f.require_group(split)

        vlen_dtype = h5py.special_dtype(vlen=np.int32)
        audio_ds = grp.get("audio") or grp.create_dataset("audio", shape=(0,), maxshape=(None,), dtype=vlen_dtype)
        text_ds = grp.get("text") or grp.create_dataset("text", shape=(0,), maxshape=(None,), dtype=vlen_dtype)
        length_ds = grp.get("length") or grp.create_dataset("length", shape=(0,), maxshape=(None,), dtype=np.int32)

        n = len(audio_tokens_batch)
        audio_ds.resize(audio_ds.shape[0] + n, axis=0)
        text_ds.resize(text_ds.shape[0] + n, axis=0)
        length_ds.resize(length_ds.shape[0] + n, axis=0)

        for i in range(n):
            audio_array = np.array(audio_tokens_batch[i], dtype=np.int32).flatten()  # [n_codebooks * seq_len]
            text_array = np.array(text_tokens_batch[i], dtype=np.int32)

            seq_len = audio_array.shape[0] // AUDIO_NUM_CODEBOOKS
            total_len = seq_len + len(text_array) + 1  # +1 for EOS frame

            audio_ds[-n + i] = audio_array
            text_ds[-n + i] = text_array
            length_ds[-n + i] = total_len


def get_num_existing_samples(file_path, split):
    """Return the number of existing samples in the HDF5 file for the given split, using the 'length' dataset."""
    try:
        with h5py.File(file_path, "r") as f:
            if split in f and "length" in f[split]:
                return f[split]["length"].shape[0]
            return 0
    except Exception:
        return 0


def tokenize_and_store(dataset_split, output_path, split_name_hdf5, audio_tokenizer, text_tokenizer, device, save_every=100, omit_speaker_id=False):
    """
    Tokenize the dataset (loaded from Hugging Face Hub) and save in HDF5 incrementally, resuming if interrupted.
    'dataset_split' is a Hugging Face Dataset object (e.g., loaded_dataset['train'])
    'split_name_hdf5' is the name to use for the group in HDF5 (e.g., "train" or "val")
    """
    n_existing = get_num_existing_samples(output_path, split_name_hdf5)
    
    # Slicing the Dataset object for resuming
    if n_existing > 0 and n_existing < len(dataset_split):
        print(f"⏩ Resuming {split_name_hdf5}: skipping {n_existing} already processed samples out of {len(dataset_split)}")
        # dataset.select(range(start, end)) is how you slice HF datasets
        dataset_to_process = dataset_split.select(range(n_existing, len(dataset_split)))
    elif n_existing >= len(dataset_split) and len(dataset_split) > 0 :
        print(f"✅ {split_name_hdf5}: All {len(dataset_split)} samples already processed. Skipping.")
        return
    else:
        print(f"🔄 Processing {split_name_hdf5} split: {len(dataset_split)} samples")
        dataset_to_process = dataset_split

    if len(dataset_to_process) == 0:
        if len(dataset_split) > 0: # This means all were processed
             print(f"No new samples to process for {split_name_hdf5}.")
        else: # This means the original split was empty
             print(f"Warning: {split_name_hdf5} split is empty. Skipping.")
        return

    audio_tokens_batch, text_tokens_batch = [], []

    # Iterate over the Hugging Face Dataset object
    # `row` will be a dictionary with keys like 'text', 'path' (which is now a dict itself), 'speaker' (if present)
    for row in tqdm(dataset_to_process, total=len(dataset_to_process), desc=f"Tokenizing {split_name_hdf5}"):
        # The 'path' key now holds a dictionary from the Audio feature:
        # {'path': 'cached_file_path_on_H100', 'array': numpy_array, 'sampling_rate': int}
        audio_info = row["path"] # Assuming 'path' was the audio column name
        waveform_array = audio_info["array"]
        sr = audio_info["sampling_rate"]

        # Convert numpy array from Audio feature to torch tensor
        # The array is usually mono (samples,) or stereo (channels, samples)
        # Ensure it's [1, samples] or [channels, samples] for torchaudio
        waveform = torch.from_numpy(waveform_array)
        if waveform.ndim == 1: # if it's (samples,)
            waveform = waveform.unsqueeze(0) # make it (1, samples)

        # Handle optional timestamps for slicing (if 'start' and 'end' are columns in your dataset)
        # This slicing happens *after* the full audio is loaded by `datasets`
        if "start" in row and "end" in row and row["start"] is not None and row["end"] is not None:
            frame_offset = int(row["start"] * sr)
            num_frames = int((row["end"] - row["start"]) * sr)
            
            # Slicing the waveform tensor
            # waveform is [channels, total_samples]
            waveform = waveform[:, frame_offset : frame_offset + num_frames]


        # Resample audio
        if sr != MIMI_SAMPLE_RATE:
            # Resample works on [channels, time] or [time]
            # torchaudio.functional.resample expects input tensor, orig_freq, new_freq
            # If waveform is [1, samples], squeeze might be needed if resample expects [samples] for mono,
            # but generally [channels, time] is fine.
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=MIMI_SAMPLE_RATE)
        
        # Ensure waveform is in the expected shape for the audio_tokenizer: [batch, channels, time]
        # Your original code had .unsqueeze(0).unsqueeze(0) after loading.
        # Here, waveform is already [channels, time]. If tokenizer needs [1,1,T], add batch dim.
        # Let's assume audio_tokenizer.encode expects [batch_size, num_channels, seq_len]
        # and our waveform is [num_channels, seq_len] after resample.
        waveform = waveform.unsqueeze(0).to(device) # Add batch dimension: [1, channels, seq_len]

        # Tokenize
        # audio_tokenizer.encode might expect a specific shape, adjust waveform if necessary
        # E.g., if it expects [B, T] for mono, and waveform is [1,1,T], then waveform.squeeze(1)
        audio_tokens = audio_tokenizer.encode(waveform)[0].tolist()  # shape: [n_codebooks, seq_len]
        
        speaker = row.get("speaker", 999) # Get speaker ID if 'speaker' column exists in dataset
        text_content = row['text']
        text = f"[{speaker}]{text_content}" if not omit_speaker_id else text_content
        text_tokens = text_tokenizer.encode(text)

        # Accumulate batch
        audio_tokens_batch.append(audio_tokens)
        text_tokens_batch.append(text_tokens)

        if len(audio_tokens_batch) >= save_every:
            append_to_hdf5(output_path, split_name_hdf5, audio_tokens_batch, text_tokens_batch)
            audio_tokens_batch, text_tokens_batch = [], []

    # Final flush
    if audio_tokens_batch:
        append_to_hdf5(output_path, split_name_hdf5, audio_tokens_batch, text_tokens_batch)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    try:
        text_tokenizer, audio_tokenizer = load_tokenizers(device=device)
    except Exception as e:
        print(f"FATAL: Error loading tokenizers: {e}")
        print("Ensure utils.py, .env, CSM_REPO_PATH, and all dependencies are correctly set up.")
        exit(1)

    print(f"Loading training data from Hub: {args.train_repo_id}, split: {args.train_split_name}")
    # REMOVED use_auth_token=True
    train_dataset_split = load_dataset(args.train_repo_id, split=args.train_split_name) 
                                                                                    
    tokenize_and_store(
        train_dataset_split, args.output, "train",
        audio_tokenizer, text_tokenizer, device, args.save_every, args.omit_speaker_id
    )

    print(f"Loading validation data from Hub: {args.val_repo_id}, split: {args.val_split_name}")
    # REMOVED use_auth_token=True
    val_dataset_split = load_dataset(args.val_repo_id, split=args.val_split_name)
                                                                                
    tokenize_and_store(
        val_dataset_split, args.output, "val",
        audio_tokenizer, text_tokenizer, device, args.save_every, args.omit_speaker_id
    )

    print(f"\n✅ Done. Tokenized data saved to: {args.output}")