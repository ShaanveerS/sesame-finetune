from transformers import MimiModel
import torch
import math
import numpy as np

from torch.nn.utils.rnn import pad_sequence


def get_target_length(arr: np.ndarray, sampling_rate: int) -> int:
    """Calculates the target length for Mimi codes based on audio array length and sampling rate."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Input array must be a numpy array, got {type(arr)}")
    if arr.ndim == 0: # Handle potential scalar arrays if they occur
        return 0
    # Use shape for numpy array dimension size
    return math.ceil(arr.shape[-1] / (sampling_rate / 12.5))


def batch_wav_encoder(batch_dict, model: MimiModel):
    """
    For use with HF Datasets in batched mode.
    Expects audio data as numpy arrays from datasets.Audio.
    """
    batch = batch_dict["audio"]
    # Pass the actual sampling rate for each sample
    target_lengths = [get_target_length(sample["array"], sample["sampling_rate"]) for sample in batch]

    # Single batch processing
    # Convert numpy arrays to tensors before padding
    # Also convert to float32, as models usually expect this dtype
    tensor_batch = [torch.from_numpy(sample["array"]).float() for sample in batch]
    padded_batch = pad_sequence(
        tensor_batch, batch_first=True
    ).unsqueeze(1)  # (batch, 1, time)

    # Ensure padding mask is created correctly for tensors
    padding_mask = (padded_batch != 0).float() # Use float for mask if model expects it

    with torch.no_grad():
        enc_out_cuda = model.encode(
            padded_batch.to("cuda"), padding_mask=padding_mask.to("cuda")
        )
    enc_out = enc_out_cuda.audio_codes.clone().cpu()
    del enc_out_cuda
    torch.cuda.empty_cache()

    # Process outputs
    chunked = torch.unbind(enc_out, dim=0)
    outputs = [t[:, :l] for t, l in zip(chunked, target_lengths)]

    return {"codes": outputs}