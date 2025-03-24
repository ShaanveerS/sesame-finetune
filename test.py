from modeling.models import Model, ModelArgs
from modeling.generator import Generator
import soundfile as sf
import torch
import json

def main():
    MODEL_FOLDER = "./inits/csm-1b-expresso/"
    with open(f"{MODEL_FOLDER}/config.json", 'r') as f:
        model_config = json.load(f)

    model = Model(ModelArgs(**model_config))
    state_dict = torch.load(f"{MODEL_FOLDER}/final_model.pt")
    model.load_state_dict(state_dict, strict=True)
    model.to(device="cuda", dtype=torch.bfloat16)
    model.decoder = torch.compile(model.decoder)
    gen = Generator(model, tokenizer_name="./inits/csm-1b-expresso")
    print(gen._text_tokenizer.encode("<|sad|>"))


    out = torch.concat([frame for frame in gen.generate("<|laughing|>There'll be a funnel cloud Monday, but it'll be mostly sunny Tuesday.", 0, context=[])])

    sf.write("output.wav", out.cpu().numpy(), samplerate=gen.sample_rate)

if __name__ == "__main__":
    main()
