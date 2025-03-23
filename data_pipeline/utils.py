from transformers import MimiModel
import torch
import math

from torch.nn.utils.rnn import pad_sequence


def get_target_length(arr: torch.Tensor, SAMPLING_RATE: int = 24_000) -> int:
    return math.ceil(arr.size(-1) / (SAMPLING_RATE / 12.5))


def batch_wav_encoder(batch_dict, model: MimiModel):
    """
    For use with HF Datasets in batched mode.
    Assumes WAV is already resampled to SAMPLING_RATE and in audio column.
    """
    batch = batch_dict["audio"]
    target_lengths = [get_target_length(sample["array"]) for sample in batch]

    # Single batch processing
    padded_batch = pad_sequence(
        [sample["array"] for sample in batch], batch_first=True
    ).unsqueeze(1)  # (batch, 1, time)

    padding_mask = (padded_batch != 0).float()

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