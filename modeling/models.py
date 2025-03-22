from dataclasses import dataclass

from einops import rearrange
import torch
import torch.nn as nn
import torchtune
from torchtune.generation._generation import get_causal_mask_from_padding_mask
from huggingface_hub import PyTorchModelHubMixin
from torchtune.models import llama3_2


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(
    probs,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/SesameAILabs/csm",
    pipeline_tag="text-to-speech",
    license="apache-2.0",
):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, self.audio_head.device),
        )

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(
                max_batch_size,
                dtype,
                decoder_max_seq_len=self.config.audio_num_codebooks,
            )

        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(self.backbone.max_seq_len, device),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        skip_audio_proj: bool = False,
    ) -> torch.Tensor:
        """
        Returns raw acoustic token hidden states
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        curr_backbone_mask = get_causal_mask_from_padding_mask(key_padding_mask)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, mask=curr_backbone_mask).to(
            dtype=dtype
        )

        c0_logits = self.codebook0_head(h)
        print(c0_logits.shape)
        x = self.projection(h)
        # Skip final token
        print(f"TOKENS SHAPE: {tokens.shape}")
        codebooks = tokens[:, :, :-2]
        print(f"CODEBOOKS BEFORE FLIP") 
        print(codebooks[0, :32, :])
        offset = torch.arange(
            0, 
            (self.config.audio_num_codebooks - 1) * self.config.audio_vocab_size, 
            self.config.audio_vocab_size, 
            device=x.device
        )

        codebooks = codebooks + offset

        # Unlike Moshi, we keep both the first hidden state and codebook 0
        codebook_embeddings_full = self.audio_embeddings(codebooks)
        # Project to decoder dimension. See decoder_h below
        codebook_embeddings = self.projection(codebook_embeddings_full)

        x = torch.cat([x.unsqueeze(-2), codebook_embeddings], dim=-2)

        b, s = x.size(0), x.size(2)
        # Flip the sequence to
        # see https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice "compute amortization"
        x = rearrange(x, 'b n s d -> (b n) s d')
        # print(f"FLIPPED: {x.shape}")

        # raise ValueError("TEST")
        # Remove padded positions
        codebooks = rearrange(codebooks, 'b n s -> (b n) s')
        print(f"CODEBOOKS AFTER FLIP: {codebooks.shape}")
        # TODO handle full text-only batches - mask will be all True
        codebook_mask = (codebooks == 0).all(dim=-1)

        # get all audio positions
        x_bs, x_len = x.size(0), x.size(1)
        indices = torch.arange(x_bs, device=x.device)[~codebook_mask]
        x = torch.index_select(x, 0, indices)

        assert x_len == self.config.audio_num_codebooks

        # create causal mask
        # curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, indices)[None, None, :, :]
        # print(curr_decoder_mask.shape)
        
        # TODO this is almost certainly wrong
        # TODO add compute amortization here

        for layer in self.decoder.layers:
            # TODO gradient checkpointing probably needs to be added
            x = layer(x)

        print(f"OUT SHAPE: {x.shape}")
        # Forward pass in parallel
        # TODO codebook logits

        # TODO make this configurable
        output_dim = self.config.audio_vocab_size if not skip_audio_proj else 1024
        if not skip_audio_proj:
            print(self.audio_head.shape)
            print(x.shape)
            # Per Sesame authors, losses for decoder are only computed on codebooks 1 and up
            x = torch.einsum('bch, cho -> bco', x[:, 1:, :], self.audio_head)
            

        buffer_len = x_len - 1
        buffer = torch.zeros(
            x_bs,
            buffer_len,
            output_dim,
            dtype=dtype,
            device=x.device
        )
        print(buffer.shape)
        # TODO almost certainly wrong, validate tomorrow
        buffer.scatter_(
            0,
            indices.view(-1, 1, 1).expand(-1, buffer_len, x.size(-1)),
            x,
        )
        buffer = rearrange(buffer, '(b n) s d -> b n s d', b=b, s=buffer_len, d=output_dim)

        if skip_audio_proj:
            return buffer

        buffer = torch.cat([c0_logits.unsqueeze(-2), buffer], dim=-2)
        raise ValueError("TEST")
        return buffer


    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(
            dtype=dtype
        )

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            ).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
