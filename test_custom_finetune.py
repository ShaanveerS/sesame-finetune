import argparse
from modeling.models import Model, ModelArgs
from modeling.generator import Generator # Generator uses the correct tokenizer loading logic
import soundfile as sf
import torch
import json
import os

parser = argparse.ArgumentParser(description="Test Custom CSM model fine-tune")
# --- MODIFIED ---
parser.add_argument("--checkpoint_path", type=str, default="/home/shaan/Projects/Dataset/checkpoints/tara_model.pt", help="Path to the fine-tuned final_model.pt file.")
parser.add_argument("--config_path", type=str, default="/home/shaan/Projects/Dataset/csm_finetune/inits/csm-1b-expresso/config.json", help="Path to the model config JSON (should be the base config).")
parser.add_argument("--tokenizer_path", type=str, default="/home/shaan/Projects/Dataset/csm_finetune/inits/csm-1b-expresso", help="Path to the base tokenizer directory.")
# --- Speaker ID likely still relevant ---
parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID (integer used during training).")
parser.add_argument("--text", type=str, default="so yeah, exactly. we provide uh fifteen appointments or your money back. it's our guarantee.", help="Text to generate")
# --- REMOVED Style argument ---
# parser.add_argument("--style", type=str, ...)

def main():
    args = parser.parse_args()

    # --- MODIFIED: Load config from specified path ---
    if not os.path.exists(args.config_path):
         print(f"Error: Config file not found at {args.config_path}")
         return
    print(f"Loading model config from {args.config_path}")
    with open(args.config_path, 'r') as f:
        model_config = json.load(f)

    # --- MODIFIED: Load fine-tuned state_dict ---
    if not os.path.exists(args.checkpoint_path):
         print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
         return
    print(f"Loading fine-tuned model state_dict from {args.checkpoint_path}")
    model = Model(ModelArgs(**model_config))
    # Load the weights onto CPU first in case of GPU memory issues during loading
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True) # Use strict=True to ensure architecture matches
    model.to(device="cuda", dtype=torch.bfloat16) # Move to GPU after loading
    model.eval() # Set to evaluation mode
    # You might still want to compile the decoder for inference speed
    # model.decoder = torch.compile(model.decoder) # Optional

    # --- MODIFIED: Initialize generator with BASE tokenizer path ---
    print(f"Initializing generator with tokenizer from {args.tokenizer_path}")
    # Generator's __init__ loads the tokenizer correctly
    gen = Generator(model, tokenizer_name=args.tokenizer_path)

    # --- MODIFIED: Format text WITHOUT style token ---
    input_text = args.text
    print(f"Generating audio for text: '{input_text}' with speaker_id: {args.speaker_id}")
    # Context is empty for simple generation
    out_frames = []
    try:
         # The generator internally uses the correct format: f"[{speaker}]{text}"
         for frame in gen.generate(text=input_text, speaker=args.speaker_id, context=[]):
              out_frames.append(frame)
         out = torch.cat(out_frames)
    except Exception as e:
         print(f"Error during generation: {e}")
         return

    output_filename = "finetuned_output.wav"
    print(f"Saving audio to {output_filename}")
    sf.write(output_filename, out.cpu().numpy(), samplerate=gen.sample_rate)
    print("Done.")

if __name__ == "__main__":
    main()