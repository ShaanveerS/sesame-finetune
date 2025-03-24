# CSM Finetuning: Expresso proof of concept

> [!WARNING] 
> This repo is a work in progress and is not ready for general use.
> I've verified that training works for expresso dataset as shown here, but essential features like a proper data cleaning script, LR scheduler, LoRA, checkpointing, and schema are missing.
> 
> Do NOT use this repo unless you are a dev and comfortable doing some heavy editing to make it work with your dataset. We will eventually provide a proper script and set of instructions. No support will be provided yet unless you are a contributor.

# Usage

Clone the repo and use `uv`:

```shell
uv sync
uv pip install -e .
```

Then create your dataset for [Expresso](https://huggingface.co/datasets/ylacombe/expresso):

```bash
cd data_pipeline
# Should be pretty short: ~2-3 min on 4090
uv run convert_expresso.py
```

Run the `data_pipeline/tokenize_expresso.ipynb` notebook.

Finally, edit the `config/train_expresso.toml` file with your training requirements, then run:

```shell
cd training_harness
uv run main.py
```

If you want to try out your model, use `test_expresso.py`:

```shell
uv run test_expresso.py \
    --speaker_id 0 \ # Speakers 1-4 from Expresso dataset
    --style default \ # Supports: default, happy, laughing, sad, whisper, emphasis, enunciated, confused
    --text "There'll be a funnel cloud Monday, but it'll be mostly sunny Tuesday."
```

Thanks to [nytopop](https://github.com/nytopop/illu) for the repo base.
