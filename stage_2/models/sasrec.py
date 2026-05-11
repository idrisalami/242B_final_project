"""SASRec-original (Kang & McAuley, 2018) — small causal-attention transformer.

Initialized from Stage 1's ALS item embeddings; fine-tuned end-to-end with
causal LM training. See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §3.
"""

import torch
import torch.nn as nn


class _Block(nn.Module):
    """One pre-LN Transformer block: causal MHA + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=False,            # we pass attn_mask explicitly
        )
        x = x + self.drop(a)
        x = x + self.ffn(self.ln2(x))
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        ffn_dim: int = 256,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Causal mask cached at construction time
        causal = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[self.pad_id].zero_()
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def load_als(self, als_factors: torch.Tensor):
        """Load ALS factors into rows [1:]. Row 0 (PAD) stays zero.

        als_factors: shape (vocab_size - 1, d_model), float32.
        """
        v_minus_1, d = als_factors.shape
        assert v_minus_1 == self.item_emb.weight.shape[0] - 1, \
            f"ALS rows ({v_minus_1}) must equal vocab_size - 1 ({self.item_emb.weight.shape[0] - 1})"
        assert d == self.item_emb.weight.shape[1], \
            f"ALS dim ({d}) must equal d_model ({self.item_emb.weight.shape[1]})"
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.item_emb.weight[1:].copy_(als_factors)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) int64 padded with pad_id=0. Returns (B, L, d_model)."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.input_drop(self.input_ln(x))

        pad_mask = (seq == self.pad_id)        # (B, L) — True where PAD
        attn_mask = self.causal_mask[:L, :L]   # (L, L) — True where blocked

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=pad_mask)
        return self.final_ln(x)

    def param_groups(self):
        """Split trainable params into (embedding, others) for discriminative fine-tuning."""
        embed_params = [self.item_emb.weight]
        other_params = [
            p for n, p in self.named_parameters()
            if n != "item_emb.weight" and p.requires_grad
        ]
        return embed_params, other_params
