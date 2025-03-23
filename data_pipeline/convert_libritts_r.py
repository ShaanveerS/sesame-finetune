from datasets import load_dataset, Audio
import datasets
from transformers import MimiModel
import argparse
from tqdm import tqdm
from pathlib import Path
from data_pipeline.utils import batch_wav_encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="encoded_libritts",
        help="Directory where each split folder will be saved",
    )
    args = parser.parse_args()

    global SAMPLING_RATE, model
    SAMPLING_RATE = 24_000

    # Load model
    model = MimiModel.from_pretrained("kyutai/mimi").to("cuda")

    # The splits we want to process
    all_splits = [
        "dev.clean",
        "test.clean",
        "train.clean.100",
        "train.clean.360",
    ]

    # Load the source dataset with streaming
    dataset_dict = load_dataset("mythicinfinity/libritts_r", "clean", streaming=True)

    output_dir = Path(args.output)

    for split in all_splits:
        print(f"\nProcessing {split}...")

        # We'll stream the dataset, with the audio column cast to the correct sample rate
        streamed_split = dataset_dict[split]
        streamed_split = streamed_split.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )
        streamed_split = streamed_split.with_format("torch")

        # We'll accumulate encoded rows in memory
        # (If you truly can't fit them all, you'd reintroduce sharding.)
        encoded_rows = []
        print("Encoding in batches...")

        for batch_out in tqdm(
            streamed_split.map(
                batch_wav_encoder,
                batched=True,
                batch_size=24,
                remove_columns=["audio"],
            ),
            desc=f"Encoding {split}",
        ):
            encoded_rows.append(batch_out)

        new_data = datasets.Dataset.from_list(encoded_rows)

        # Save to disk in a subfolder named after the split
        split_folder = output_dir / split
        print(f"Saving {split} to {split_folder}...")
        split_folder.mkdir(parents=True, exist_ok=True)
        new_data.save_to_disk(str(split_folder))

        print(f"Finished {split}")

    print("\nAll splits processed. Done!")


if __name__ == "__main__":
    main()
