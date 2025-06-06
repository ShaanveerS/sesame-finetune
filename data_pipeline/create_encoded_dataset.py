# data_pipeline/create_encoded_dataset.py

import argparse
import pandas as pd
from pathlib import Path
import torch
# Use numpy for checking audio array validity robustly
import numpy as np
from datasets import Dataset, Audio, Value, Features, Sequence
from transformers import MimiModel
import logging
import os
import soundfile as sf # Import soundfile to explicitly check audio data length

# Import the exact encoder function used by the original scripts
try:
    # Assumes this script is run from the repo root or utils is in path
    from utils import batch_wav_encoder
    # If batch_wav_encoder import works, we need to decide where SAMPLING_RATE comes from.
    # For now, let's assume the fallback is always used if SAMPLING_RATE isn't imported.
    # This might need adjustment if utils *should* provide it elsewhere.
    BATCH_ENCODER_SAMPLING_RATE = 24_000 # Temporarily set fallback here as well
except ImportError:
    logging.error("Could not import batch_wav_encoder from data_pipeline.utils. Make sure PYTHONPATH is set or run from repo root.")
    # Define fallbacks or raise error if import fails
    BATCH_ENCODER_SAMPLING_RATE = 24_000 # Fallback, ensure it matches Mimi's requirement
    # Define batch_wav_encoder here if needed as a fallback, or raise the error
    raise # Keep raising for now as the original did

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Use the sampling rate defined/imported from utils
SAMPLING_RATE = BATCH_ENCODER_SAMPLING_RATE
MIMI_MODEL_NAME = "kyutai/mimi"
DEFAULT_STYLE = "default" # Assign a default style if your data doesn't have it

# --- Helper Function to Check Audio Validity ---
def check_audio_validity(example):
    """
    Checks if the audio file path in the example corresponds to a valid, non-empty audio file.
    Uses soundfile for a direct check before datasets loads it fully.

    Args:
        example (dict): A dataset example containing 'audio_filepath'.

    Returns:
        dict: A dictionary {'is_valid': bool} indicating validity.
    """
    is_valid = False
    audio_path = example.get('audio_filepath')
    path_for_logging = example.get('relative_wav_path', audio_path) # Prefer relative path for logging

    if not audio_path or not Path(audio_path).is_file():
        logging.warning(f"Audio file path missing or file not found: {path_for_logging}")
        return {'is_valid': False}

    try:
        # Use soundfile to quickly check if the file can be opened and has data
        # Read only the first frame to check for validity without loading everything
        with sf.SoundFile(audio_path, 'r') as f:
            # Check if there are frames (data) in the file
            if f.frames > 0:
                is_valid = True
            else:
                logging.warning(f"Audio file contains no data (0 frames): {path_for_logging}")
                is_valid = False

    except Exception as e:
        # Catch errors during file opening (e.g., corrupted file, unsupported format)
        logging.warning(f"Failed to read audio file (may be corrupted or empty): {path_for_logging}. Error: {e}")
        is_valid = False

    return {'is_valid': is_valid}


def main():
    parser = argparse.ArgumentParser(description="Encode segmented audio data using Mimi for CSM fine-tuning (mimics convert_expresso.py).")
    parser.add_argument("--metadata_file", default="/home/shaan/Projects/Dataset/segmented_data/metadata.csv", help="Path to the input metadata file (e.g., metadata.csv, format: relative_wav_path|text|speaker_id).")
    parser.add_argument("--wav_directory", default="/home/shaan/Projects/Dataset/segmented_data/wavs", help="Path to the directory containing the segmented WAV files (used for logging/context if needed, primarily relies on paths in metadata).")
    parser.add_argument("--output_directory", default="/home/shaan/Projects/Dataset/datasets/encoded_custom_data", help="Directory to save the encoded Hugging Face dataset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for audio encoding (adjust based on GPU memory).")
    parser.add_argument("--num_proc", type=int, default=None, help="Number of processes for mapping (if desired, None for sequential).")

    args = parser.parse_args()

    logging.info(f"Starting dataset creation from {args.metadata_file}")
    metadata_path = Path(args.metadata_file)
    # wav_dir = Path(args.wav_directory) # wav_dir isn't strictly needed if metadata has full/relative paths
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Metadata ---
    try:
        df = pd.read_csv(metadata_path, sep='|', header=None, names=['relative_wav_path', 'text', 'speaker_id'], dtype={'speaker_id': str})
        # Construct absolute path relative to the metadata file's location
        metadata_dir = metadata_path.parent
        df['audio_filepath'] = df['relative_wav_path'].apply(lambda x: str((metadata_dir / x).resolve()))
        df['style'] = DEFAULT_STYLE
        logging.info(f"Loaded metadata with {len(df)} rows.")

        # Initial file existence check (Quick check, doesn't guarantee valid audio)
        initial_count = len(df)
        df = df[df['audio_filepath'].apply(lambda x: Path(x).is_file())]
        if len(df) < initial_count:
            logging.warning(f"Filtered out {initial_count - len(df)} rows due to missing files (based on path existence check).")

        if df.empty:
             logging.error("No valid data remaining after checking file existence. Exiting.")
             return

    except Exception as e:
        logging.error(f"Error loading or processing metadata file '{metadata_path}': {e}", exc_info=True)
        return

    # --- 2. Create Initial Hugging Face Dataset Structure (Paths Only) ---
    # We will load audio later, but define the structure now.
    # Use file paths for initial creation.
    data_dict = df.to_dict(orient='list')
    initial_features_paths_only = Features({
        'audio_filepath': Value('string'), # Store the path first
        'text': Value('string'),
        'speaker_id': Value('string'),
        'style': Value('string'),
        'relative_wav_path': Value('string') # Keep original path if needed
    })

    try:
        dataset = Dataset.from_dict(data_dict, features=initial_features_paths_only)
        logging.info(f"Created initial Dataset structure with {len(dataset)} entries (paths only).")
    except Exception as e:
        logging.error(f"Error creating initial Hugging Face Dataset structure: {e}", exc_info=True)
        return

    # --- 3. Filter out entries with invalid/empty audio files ---
    logging.info("Validating audio files (checking readability and non-emptiness)...")
    try:
        # Add a temporary column 'is_valid' using the helper function
        dataset_with_validity = dataset.map(
            check_audio_validity,
            num_proc=args.num_proc,
            keep_in_memory=False # Keep memory usage low
        )

        # Filter the dataset based on the 'is_valid' column
        valid_dataset = dataset_with_validity.filter(
            lambda example: example['is_valid'],
            num_proc=args.num_proc,
            keep_in_memory=False
        )

        # Remove the temporary columns
        valid_dataset = valid_dataset.remove_columns(['is_valid', 'audio_filepath'])

        num_removed = len(dataset) - len(valid_dataset)
        if num_removed > 0:
            logging.warning(f"Removed {num_removed} entries due to invalid or empty audio files.")
        logging.info(f"Proceeding with {len(valid_dataset)} valid audio entries.")

        if len(valid_dataset) == 0:
            logging.error("No valid audio files found after validation. Exiting.")
            return

        # Now, cast the 'audio' column correctly using the validated file paths
        # Recreate the data_dict from the filtered dataset for casting
        valid_data_dict = valid_dataset.to_dict()
        # We need the absolute paths again, which were in the original 'audio_filepath'
        # Let's reconstruct them (this is slightly inefficient, alternative: don't remove audio_filepath until end)
        # Re-add absolute paths based on relative path for the final dataset creation
        valid_data_dict['audio'] = [str((metadata_dir / rel_path).resolve()) for rel_path in valid_data_dict['relative_wav_path']]


        final_features = Features({
            'audio': Audio(sampling_rate=SAMPLING_RATE), # NOW define Audio feature for loading
            'text': Value('string'),
            'speaker_id': Value('string'),
            'style': Value('string'),
            'relative_wav_path': Value('string')
        })

        # Create the final dataset ready for encoding
        dataset = Dataset.from_dict(valid_data_dict, features=final_features)
        logging.info("Created final Dataset with Audio feature for valid files.")


    except Exception as e:
        logging.error(f"Error during audio validation or final dataset creation: {e}", exc_info=True)
        return


    # --- 4. Load Mimi Model ---
    try:
        logging.info(f"Loading Mimi model: {MIMI_MODEL_NAME}")
        model = MimiModel.from_pretrained(MIMI_MODEL_NAME)
        if torch.cuda.is_available():
            model.to("cuda")
            logging.info("Mimi model moved to CUDA.")
        else:
            logging.warning("CUDA not available, Mimi model running on CPU.")
        model.eval()
    except Exception as e:
        logging.error(f"Error loading Mimi model: {e}", exc_info=True)
        return

    # --- 5. Encode Audio using map (like convert_expresso.py) ---
    logging.info("Starting audio encoding using map...")
    try:
        # The .map call will now:
        # - Load audio from the 'audio' column (which uses the paths)
        # - Resample to SAMPLING_RATE (defined in Audio feature)
        # - Group rows into batches
        # - Pass batches {'audio': [ {'path': None, 'array': tensor1, 'sampling_rate': SR}, ... ], ...} to the lambda
        # - Our lambda calls batch_wav_encoder which expects {'audio': [ {'array': tensor1}, ... ]} format
        encoded_ds = dataset.map(
            lambda batch: batch_wav_encoder(batch, model), # Use the imported function directly
            batched=True,
            batch_size=args.batch_size,
            remove_columns=["audio"], # Remove the loaded audio array data after encoding
            num_proc=args.num_proc, # Use multiprocessing if specified
            keep_in_memory=False, # Safer for potentially large datasets
            # features=output_features # Optional: Explicitly define output features if needed
        )
        logging.info("Finished audio encoding map.")

    except Exception as e:
        logging.error(f"Error during batch encoding map: {e}", exc_info=True)
        # If map fails here, it's likely an issue within batch_wav_encoder or resource limits,
        # as invalid audio files should have been filtered out.
        return

    # --- 6. Save Encoded Dataset ---
    try:
        logging.info(f"Saving encoded dataset to {output_dir}")
        # The 'codes' column should now exist and contain the encoded tensors
        encoded_ds.save_to_disk(str(output_dir))
        logging.info("Encoded dataset saved successfully.")
    except Exception as e:
        logging.error(f"Error saving encoded dataset: {e}", exc_info=True)
        return

    logging.info("Dataset creation complete.")


if __name__ == "__main__":
    main()