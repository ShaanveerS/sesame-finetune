from training_harness.data import collate_fn
from modeling.utils import PromptEncoder
import torch
import random

def make_fake_utterances(bsz: int):
    """
    Generates a list of fake utterances for testing.
    
    For each of the bsz items:
      - Randomly determine a number of words between 0 and 25.
      - Pick that many words (with replacement) from a fake dictionary.
      - Join the words with spaces to form an utterance.
      
    Returns:
      List[str]: A list of bsz utterances.
    """
    fake_dict = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", 
        "hotel", "india", "juliet", "kilo", "lima", "mike", "november", 
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", 
        "victor", "whiskey", "xray", "yankee", "zulu"
    ]
    
    utterances = []
    for _ in range(bsz):
        num_words = random.randint(0, 25)
        # If num_words is 0, you'll end up with an empty utterance.
        words = [random.choice(fake_dict) for _ in range(num_words)]
        utterance = " ".join(words)
        utterances.append(utterance)
        
    return utterances


# yolo, let the llm do it
def make_fake_mimi_codes(bsz: int):
    """
    Generates a list of fake RVQ codes for testing.
    
    For each of the bsz items:
      - Generate a random sequence length between 1 and 255.
      - Create a tensor of shape (32, seqlen) with random ints in [0, 2051).
      
    Returns:
      List[torch.Tensor]: A list containing bsz tensors.
    """
    codes = []
    for _ in range(bsz):
        # Pick a random sequence length between 1 and 255 (inclusive)
        seqlen = torch.randint(1, 256, (1,)).item()
        # Create a (32, seqlen) tensor with random "codes" from 0 to 2050.
        code_tensor = torch.randint(0, 2051, (32, seqlen))
        codes.append(code_tensor)
    return codes



def test_collate_fn():
    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    prompt_encoder = PromptEncoder()
    fake_codes = make_fake_mimi_codes(24)
    fake_utterances = make_fake_utterances(24)

    fake_ds = []
    for utterance, codes in zip(fake_utterances, fake_codes):
        audio_tokens, audio_masks = prompt_encoder._tokenize_audio(codes)
        text_tokens, text_masks = prompt_encoder._tokenize_text_segment(utterance, speaker=0)
        
        fake_ds.append({
            "ground_truth": torch.cat([text_tokens, audio_tokens], dim=0),
            "ground_truth_masks": torch.cat([text_masks, audio_masks], dim=0)
        })

    batch = collate_fn(fake_ds)

    # Test that all tokens (minus the last one) of each sample are in batch["tokens"]
    for i, sample in enumerate(fake_ds):
        sample_tokens = sample["ground_truth"][:-1]  # Exclude the last token
        assert torch.all(batch["tokens"][i, :len(sample_tokens)] == sample_tokens), \
            f"Mismatch in tokens for sample {i}"

    # Test that masked positions have -100s in labels
    for i, sample in enumerate(fake_ds):
        seq_len = sample["ground_truth"].shape[0] - 1
        # Get the mask for the shifted sequence (next tokens)
        mask = sample["ground_truth_masks"][1:, :-1].all(dim=1)
        # Check that masked positions have -100s
        assert torch.all(batch["labels"][i, :seq_len][~mask] == -100), \
            f"Masked positions should have -100s in labels for sample {i}"
        # Check that unmasked positions don't all have -100s
        if mask.any():
            assert not torch.all(batch["labels"][i, :seq_len][mask] == -100), \
                f"Unmasked positions should not all be -100s for sample {i}"

    # Test shapes
    B = len(fake_ds)
    max_len = max(sample['ground_truth'].shape[0] - 1 for sample in fake_ds)
    assert batch['tokens'].shape == (B, max_len, 33), 'Wrong tokens shape'
    assert batch['tokens_masks'].shape == (B, max_len, 33), 'Wrong tokens_masks shape'
    assert batch['labels'].shape == (B, max_len, 32), 'Wrong labels shape'
    assert batch['pad_mask'].shape == (B, max_len), 'Wrong pad_mask shape'

    # Test padding
    for i, sample in enumerate(fake_ds):
        seq_len = sample['ground_truth'].shape[0] - 1
        # Check that tokens are padded with zeros
        assert torch.all(batch['tokens'][i, seq_len:] == 0), f'Wrong padding in tokens for sample {i}'
        # Check pad_mask correctly marks real vs padded positions
        assert torch.all(batch['pad_mask'][i, :seq_len] == True), f'pad_mask should be True for real positions in sample {i}'
        assert torch.all(batch['pad_mask'][i, seq_len:] == False), f'pad_mask should be False for padded positions in sample {i}'
        
    # Test tokens_masks
    for i, sample in enumerate(fake_ds):
        seq_len = sample['ground_truth'].shape[0] - 1
        # Check masks match input
        assert torch.all(batch['tokens_masks'][i, :seq_len] == sample['ground_truth_masks'][:-1]), \
            f'tokens_masks does not match input masks for sample {i}'
        # Check padded positions have mask=False
        assert torch.all(batch['tokens_masks'][i, seq_len:] == False), \
            f'Padded positions should have mask=False for sample {i}'



