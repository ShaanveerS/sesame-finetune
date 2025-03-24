import argparse
from modeling.models import Model, ModelArgs
from modeling.generator import Generator
import soundfile as sf
import torch
import json

parser = argparse.ArgumentParser(description="Test Expresso model fine-tune")
parser.add_argument("--speaker_id", type=int, choices=range(4), default=0, help="Speaker ID (0-3)")
parser.add_argument("--text", type=str, default="There'll be a funnel cloud Monday, but it'll be mostly sunny Tuesday.", help="Text to generate")
parser.add_argument("--style", type=str, choices=["confused", "enunciated", "happy", "laughing", "default", "sad", "whisper", "emphasis"], default="default", help="Speaking style")

def main():
    args = parser.parse_args()
    MODEL_FOLDER = "./inits/csm-1b-expresso/"
    with open(f"{MODEL_FOLDER}/config.json", 'r') as f:
        model_config = json.load(f)

    print(f"Loading model from {MODEL_FOLDER}")
    model = Model(ModelArgs(**model_config))
    state_dict = torch.load(f"{MODEL_FOLDER}/final_model.pt")
    model.load_state_dict(state_dict, strict=True)
    model.to(device="cuda", dtype=torch.bfloat16)
    model.decoder = torch.compile(model.decoder)
    gen = Generator(model, tokenizer_name="./inits/csm-1b-expresso")

    print(f"Generating audio for {args.text}")
    out = torch.concat([frame for frame in gen.generate(f"<|{args.style}|>{args.text}", speaker=args.speaker_id, context=[])])

    print(f"Saving audio to output.wav")
    sf.write("output.wav", out.cpu().numpy(), samplerate=gen.sample_rate)

if __name__ == "__main__":
    main()
