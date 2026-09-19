import torch
import torch.nn as nn
import numpy as np

# Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expansion
            GELU(),                                         # Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Contraction
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, attention_mask=None):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # (batch, tokens, heads, head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        keys = keys.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        values = values.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        # (batch, heads, tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # -------------------------------------------------
        # Attention scores
        # -------------------------------------------------

        attn_scores = queries @ keys.transpose(2, 3)

        # -------------------------------------------------
        # 1. Causal mask
        # Prevent tokens from seeing future tokens
        # -------------------------------------------------

        causal_mask = self.mask[:num_tokens, :num_tokens].bool()

        attn_scores.masked_fill_(
            causal_mask,
            float("-inf")
        )

        # -------------------------------------------------
        # 2. Padding mask
        # Prevent attention to PAD positions
        # -------------------------------------------------

        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, None, :].bool()

            attn_scores = attn_scores.masked_fill(
                ~key_padding_mask,
                float("-inf")
            )

        # -------------------------------------------------
        # Convert scores into probabilities
        # -------------------------------------------------

        attn_weights = torch.softmax(
            attn_scores / self.head_dim ** 0.5,
            dim=-1
        )

        attn_weights = self.dropout(attn_weights)

        # -------------------------------------------------
        # Calculate context vectors
        # -------------------------------------------------

        context_vec = attn_weights @ values

        context_vec = context_vec.transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b,
            num_tokens,
            self.d_out
        )

        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, attention_mask=None):

        # Attention with shortcut
        shortcut = x

        x = self.norm1(x)

        x = self.att(
            x,
            attention_mask=attention_mask
        )

        x = self.drop_shortcut(x)

        x = x + shortcut

        # Feed-forward with shortcut
        shortcut = x

        x = self.norm2(x)

        x = self.ff(x)

        x = self.drop_shortcut(x)

        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"]
        )

        self.pos_emb = nn.Embedding(
            cfg["context_length"],
            cfg["emb_dim"]
        )

        self.drop_emb = nn.Dropout(
            cfg["drop_rate"]
        )

        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(cfg)
                for _ in range(cfg["n_layers"])
            ]
        )

        self.final_norm = LayerNorm(
            cfg["emb_dim"]
        )

        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False
        )

        # Weight tying
        self.out_head.weight = self.tok_emb.weight


    def forward_features(
        self,
        in_idx,
        attention_mask=None
    ):
        """
        Return GPT-2 hidden representations
        before the vocabulary output head.
        """

        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(
                seq_len,
                device=in_idx.device
            )
        )

        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(
                x,
                attention_mask=attention_mask
            )

        x = self.final_norm(x)

        return x


    def forward(
        self,
        in_idx,
        attention_mask=None
    ):
        """
        Standard GPT-2 language-model forward pass.
        """

        x = self.forward_features(
            in_idx,
            attention_mask=attention_mask
        )

        logits = self.out_head(x)

        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                     # (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def assign(param, value):
    # Helper to copy a numpy array into a torch Parameter
    # Handles transposition automatically for linear layer weights
    if param.ndim == 2 and value.ndim == 2 and param.shape != value.shape:
        # Assume we need to transpose the numpy weight matrix
        value = value.T
    param.data = torch.from_numpy(value.astype(np.float32))
    return param


def load_weights_into_gpt(gpt, params):
    assign(gpt.pos_emb.weight, params['wpe'])
    assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        # Attention weights: c_attn contains combined Q,K,V
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # Output projection
        assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        # Feed-forward layers
        assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        # LayerNorms
        assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    # Final LayerNorm and output head
    assign(gpt.final_norm.scale, params["g"])
    assign(gpt.final_norm.shift, params["b"])
    assign(gpt.out_head.weight, params["wte"])

def load_hf_weights_into_gpt(gpt, hf_model):
    """
    Copy pretrained Hugging Face GPT-2 weights
    into our custom GPT-2 implementation.
    """

    with torch.no_grad():

        # -------------------------------------------------
        # 1. Embeddings
        # -------------------------------------------------

        gpt.tok_emb.weight.copy_(
            hf_model.transformer.wte.weight
        )

        gpt.pos_emb.weight.copy_(
            hf_model.transformer.wpe.weight
        )


        # -------------------------------------------------
        # 2. Transformer blocks
        # -------------------------------------------------

        for i in range(len(gpt.trf_blocks)):

            custom_block = gpt.trf_blocks[i]
            hf_block = hf_model.transformer.h[i]


            # =============================================
            # Attention: Q, K, V
            # =============================================

            # Hugging Face GPT-2 stores Q, K and V
            # together in one matrix:
            #
            # [Q | K | V]
            #
            # Shape:
            # (768, 2304)

            qkv_weight = hf_block.attn.c_attn.weight

            q_w, k_w, v_w = torch.split(
                qkv_weight,
                gpt.tok_emb.embedding_dim,
                dim=1
            )

            # Hugging Face Conv1D stores its matrices
            # in the opposite orientation to nn.Linear,
            # therefore transpose them.

            custom_block.att.W_query.weight.copy_(q_w.T)
            custom_block.att.W_key.weight.copy_(k_w.T)
            custom_block.att.W_value.weight.copy_(v_w.T)


            # ---------------------------------------------
            # Q, K, V biases
            # ---------------------------------------------

            qkv_bias = hf_block.attn.c_attn.bias

            q_b, k_b, v_b = torch.split(
                qkv_bias,
                gpt.tok_emb.embedding_dim,
                dim=0
            )

            custom_block.att.W_query.bias.copy_(q_b)
            custom_block.att.W_key.bias.copy_(k_b)
            custom_block.att.W_value.bias.copy_(v_b)


            # =============================================
            # Attention output projection
            # =============================================

            custom_block.att.out_proj.weight.copy_(
                hf_block.attn.c_proj.weight.T
            )

            custom_block.att.out_proj.bias.copy_(
                hf_block.attn.c_proj.bias
            )


            # =============================================
            # Feed-forward / MLP
            # =============================================

            # Expansion:
            # 768 → 3072

            custom_block.ff.layers[0].weight.copy_(
                hf_block.mlp.c_fc.weight.T
            )

            custom_block.ff.layers[0].bias.copy_(
                hf_block.mlp.c_fc.bias
            )


            # Contraction:
            # 3072 → 768

            custom_block.ff.layers[2].weight.copy_(
                hf_block.mlp.c_proj.weight.T
            )

            custom_block.ff.layers[2].bias.copy_(
                hf_block.mlp.c_proj.bias
            )


            # =============================================
            # LayerNorm 1
            # =============================================

            custom_block.norm1.scale.copy_(
                hf_block.ln_1.weight
            )

            custom_block.norm1.shift.copy_(
                hf_block.ln_1.bias
            )


            # =============================================
            # LayerNorm 2
            # =============================================

            custom_block.norm2.scale.copy_(
                hf_block.ln_2.weight
            )

            custom_block.norm2.shift.copy_(
                hf_block.ln_2.bias
            )


        # -------------------------------------------------
        # 3. Final LayerNorm
        # -------------------------------------------------

        gpt.final_norm.scale.copy_(
            hf_model.transformer.ln_f.weight
        )

        gpt.final_norm.shift.copy_(
            hf_model.transformer.ln_f.bias
        )