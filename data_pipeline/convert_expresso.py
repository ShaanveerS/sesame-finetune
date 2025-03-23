from datasets import load_dataset, Audio
from transformers import MimiModel
import argparse
from pathlib import Path
from data_pipeline.utils import batch_wav_encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="encoded_expresso",
        help="Directory where each split folder will be saved",
    )
    args = parser.parse_args()

    global SAMPLING_RATE, model
    SAMPLING_RATE = 24_000

    # Load model
    model = MimiModel.from_pretrained("kyutai/mimi").to("cuda")

    # The splits we want to process
    # Load the source dataset
    dataset_dict = load_dataset("ylacombe/expresso")

    output_dir = Path(args.output)


    # We'll stream the dataset, with the audio column cast to the correct sample rate
    ds = dataset_dict['train']
    ds = ds.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )
    ds = ds.with_format("torch")

    # Filter out rows longer than 30 seconds
    ds = ds.filter(lambda x: x['audio']['array'].shape[0] / SAMPLING_RATE <= 30, num_proc=12)

    ds = ds.map(
        lambda batch: batch_wav_encoder(batch, model),
        batched=True,
        batch_size=24,
        remove_columns=["audio"],
    )

    # Save to disk in a subfolder named after the split
    ds.save_to_disk(output_dir)

    print("\nAll splits processed. Done!")


if __name__ == "__main__":
    main()