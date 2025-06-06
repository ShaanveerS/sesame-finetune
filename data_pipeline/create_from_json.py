# data_pipeline/create_encoded_dataset.py

import argparse
import pandas as pd
from pathlib import Path
import torch
# Use numpy for checking audio array validity robustly
import numpy as np
from datasets import Dataset, Audio, Value, Features, Sequence, load_dataset
from transformers import MimiModel
import logging
import os
import soundfile as sf # Import soundfile to explicitly check audio data length
import json # Added json import

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
DEFAULT_SPEAKER_ID = "default_speaker" # Assign a default speaker ID

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
    # --- MODIFIED: Added arguments to load from Hugging Face Hub ---
    parser.add_argument("--load_from_hf", action="store_true", help="Load the dataset directly from Hugging Face Hub instead of a local JSONL file.")
    parser.add_argument("--hf_repo_id", type=str, default="LucaZuana/brooke_top80", help="The Hugging Face repository ID to load when --load_from_hf is specified.")
    # ---
    parser.add_argument("--jsonl_metadata_file", default="/home/shaan/Projects/audiobox-aesthetics/brooke_top80.jsonl", help="Path to the input JSON Lines metadata file (format: {'text': '...', 'path': '...'}). Ignored if --load_from_hf is used.")
    parser.add_argument("--output_directory", default="/home/shaan/Projects/Dataset/datasets/encoded_custom_data", help="Directory to save the encoded Hugging Face dataset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for audio encoding (adjust based on GPU memory).")
    parser.add_argument("--num_proc", type=int, default=None, help="Number of processes for mapping (if desired, None for sequential).")

    args = parser.parse_args()

    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = None

    # --- MODIFIED: Logic to switch between loading from HF Hub and local file ---
    if args.load_from_hf:
        logging.info(f"Loading dataset from Hugging Face Hub repository: {args.hf_repo_id}")
        try:
            # Load the dataset (assumes a 'train' split exists as per the push log)
            raw_ds = load_dataset(args.hf_repo_id, split="train")
            
            # The loaded dataset from Hub has 'path', 'text', and the loaded 'audio'.
            # We need to align it with the structure expected by the rest of the script:
            # 'audio', 'text', 'speaker_id', 'style', 'relative_wav_path'
            
            def prepare_hf_dataset(batch):
                # Use the original path to create a relative path (filename)
                batch['relative_wav_path'] = [Path(p).name for p in batch['path']]
                # Add default speaker and style columns
                batch['speaker_id'] = [DEFAULT_SPEAKER_ID] * len(batch['path'])
                batch['style'] = [DEFAULT_STYLE] * len(batch['path'])
                return batch

            dataset = raw_ds.map(
                prepare_hf_dataset,
                batched=True,
                num_proc=args.num_proc,
                desc="Adding metadata columns"
            )
            
            # Remove the original 'path' column as it's now stored in 'relative_wav_path'
            dataset = dataset.remove_columns(['path'])

            # Ensure the audio is cast to the required sampling rate for the Mimi model
            dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
            
            logging.info(f"Successfully loaded and prepared {len(dataset)} entries from {args.hf_repo_id}.")
            logging.info(f"Dataset features: {dataset.features}")

        except Exception as e:
            logging.error(f"Failed to load or process dataset from Hugging Face Hub '{args.hf_repo_id}': {e}", exc_info=True)
            return

    else:
        # --- Original logic for loading from local JSONL file ---
        logging.info(f"Starting dataset creation from {args.jsonl_metadata_file}")
        jsonl_path = Path(args.jsonl_metadata_file)

        # --- 1. Load Metadata from JSONL ---
        data_list = []
        try:
            with open(jsonl_path, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue # Skip empty lines
                    try:
                        item = json.loads(line)
                        if 'path' in item and 'text' in item:
                            abs_path = Path(item['path']).resolve() # Ensure absolute path
                            if not abs_path.is_file():
                                logging.warning(f"Audio file not found (line {i+1}): {abs_path}. Skipping entry.")
                                continue

                            data_list.append({
                                'audio_filepath': str(abs_path), # Use absolute path directly
                                'text': item['text'],
                                'speaker_id': DEFAULT_SPEAKER_ID, # Use default speaker ID
                                'style': DEFAULT_STYLE,
                                'relative_wav_path': abs_path.name # Use filename as relative path
                            })
                        else:
                            logging.warning(f"Skipping invalid line {i+1} (missing 'path' or 'text'): {line}")
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line {i+1}: {line}")
                    except Exception as path_ex:
                        logging.warning(f"Error processing path on line {i+1} ('{item.get('path', 'N/A')}'): {path_ex}. Skipping entry.")


            if not data_list:
                logging.error(f"No valid data loaded from {args.jsonl_metadata_file}. Check file format and paths. Exiting.")
                return

            df = pd.DataFrame(data_list)
            logging.info(f"Loaded metadata from JSONL with {len(df)} initial valid entries.")

            if df.empty:
                 logging.error("No valid data loaded after checking file existence during parsing. Exiting.")
                 return

        except Exception as e:
            logging.error(f"Error loading or processing JSONL file '{jsonl_path}': {e}", exc_info=True)
            return

        # --- 2. Create Initial Hugging Face Dataset Structure (Paths Only) ---
        data_dict = df.to_dict(orient='list')
        initial_features_paths_only = Features({
            'audio_filepath': Value('string'),
            'text': Value('string'),
            'speaker_id': Value('string'),
            'style': Value('string'),
            'relative_wav_path': Value('string')
        })

        try:
            temp_dataset = Dataset.from_dict(data_dict, features=initial_features_paths_only)
            logging.info(f"Created initial Dataset structure with {len(temp_dataset)} entries (paths only).")
        except Exception as e:
            logging.error(f"Error creating initial Hugging Face Dataset structure: {e}", exc_info=True)
            return

        # --- 3. Filter out entries with invalid/empty audio files ---
        logging.info("Validating audio files (checking readability and non-emptiness)...")
        try:
            dataset_with_validity = temp_dataset.map(
                check_audio_validity, num_proc=args.num_proc, keep_in_memory=False
            )
            valid_dataset = dataset_with_validity.filter(
                lambda example: example['is_valid'], num_proc=args.num_proc, keep_in_memory=False
            )
            valid_dataset = valid_dataset.remove_columns(['is_valid'])

            num_removed = len(temp_dataset) - len(valid_dataset)
            if num_removed > 0:
                logging.warning(f"Removed {num_removed} entries due to invalid or empty audio files.")
            logging.info(f"Proceeding with {len(valid_dataset)} valid audio entries.")

            if len(valid_dataset) == 0:
                logging.error("No valid audio files found after validation. Exiting.")
                return

            valid_data_dict = valid_dataset.to_dict()
            valid_data_dict['audio'] = valid_data_dict.pop('audio_filepath')

            final_features = Features({
                'audio': Audio(sampling_rate=SAMPLING_RATE),
                'text': Value('string'),
                'speaker_id': Value('string'),
                'style': Value('string'),
                'relative_wav_path': Value('string')
            })
            
            # This is the final dataset object for the local-loading path
            dataset = Dataset.from_dict(valid_data_dict, features=final_features)
            logging.info("Created final Dataset with Audio feature for valid files.")

        except Exception as e:
            logging.error(f"Error during audio validation or final dataset creation: {e}", exc_info=True)
            return

    # --- At this point, `dataset` is populated from either HF or local file ---
    if dataset is None or len(dataset) == 0:
        logging.error("No data was loaded. Exiting.")
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
        encoded_ds = dataset.map(
            lambda batch: batch_wav_encoder(batch, model),
            batched=True,
            batch_size=args.batch_size,
            remove_columns=["audio"],
            num_proc=args.num_proc,
            keep_in_memory=False,
        )
        logging.info("Finished audio encoding map.")

    except Exception as e:
        logging.error(f"Error during batch encoding map: {e}", exc_info=True)
        return

    # --- 6. Save Encoded Dataset ---
    try:
        logging.info(f"Saving encoded dataset to {output_dir}")
        encoded_ds.save_to_disk(str(output_dir))
        logging.info("Encoded dataset saved successfully.")
    except Exception as e:
        logging.error(f"Error saving encoded dataset: {e}", exc_info=True)
        return

    logging.info("Dataset creation complete.")


if __name__ == "__main__":
    main()