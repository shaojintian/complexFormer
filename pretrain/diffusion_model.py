Okay, let's adapt the provided code structure to represent the LLaDA model based on the paper's description.
The key changes are:
Non-Causal Attention: The Transformer blocks will use standard Multi-Head Attention without a causal mask. Attention can flow in both directions.
Mask Prediction Objective: The model takes a partially masked sequence (xt) as input and is trained to predict the original tokens (x0) specifically at the masked positions.
Loss Calculation: The loss (typically cross-entropy) is computed only over the tokens that were masked in the input sequence.
Here's the modified code:
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import yaml
import argparse
import math # Added for RoPE

# --- Configuration ---
class LLaDAConfig(PretrainedConfig):
    """
    Configuration class for LLaDA model. Inherits from PretrainedConfig.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaDA model.
        hidden_dim (`int`, *optional*, defaults to 4096):
            Dimension of the hidden states.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the intermediate (feed-forward) layer in the Transformer block.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMS normalization layers.
        mask_token_id (`int`, *optional*, defaults to -1):
            The ID used to represent a masked token in the input sequence `xt`.
            Needs to be set based on your tokenizer.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period for Rotary Positional Embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
             The maximum sequence length that this model might ever be used with.
        **kwargs:
            Additional keyword arguments passed along to `PretrainedConfig`.
    """
    model_type = "llada"

    def __init__(
        self,
        vocab_size=32000, # Example vocab size
        hidden_dim=4096, # Example hidden dim (like Llama 7B)
        intermediate_size=11008, # Example intermediate size (like Llama 7B)
        num_hidden_layers=32, # Example layer count (like Llama 7B)
        num_attention_heads=32, # Example head count (like Llama 7B)
        dropout=0.1,
        rms_norm_eps=1e-5, # Common value for Llama-style models
        mask_token_id=0,  # IMPORTANT: Set this to your actual mask token ID!
        rope_theta=10000.0,
        max_position_embeddings=4096,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.mask_token_id = mask_token_id
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        super().__init__(**kwargs)

# --- Rotary Positional Embedding (RoPE) ---
# Based on Llama implementation details
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self._buffers["inv_freq"].device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- Model Components ---

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm without bias.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class FFN(nn.Module):
    # Standard Feed Forward Layer (often using SwiGLU in modern LLMs)
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.linear3 = nn.Linear(config.intermediate_size, config.hidden_dim, bias=False) # W_o in SwiGLU
        self.linear2 = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False) # Gate proj
        self.activation = nn.SiLU() # Swish activation

    def forward(self, x):
        # SwiGLU implementation
        x = self.linear3(self.activation(self.linear1(x)) * self.linear2(x))
        return x

class LLaDAAttention(nn.Module):
    # Multi-Head Attention specifically for LLaDA (non-causal)
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_dim:
            raise ValueError(f"hidden_dim must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.config.max_position_embeddings, base=self.config.rope_theta)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None, # Padding mask
        position_ids: torch.LongTensor | None = None,
        # LLaDA does NOT use causal masks in attention
    ) -> torch.Tensor:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len) # Use value_states just for dtype/device
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # query_states, key_states: [bsz, num_heads, q_len, head_dim]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, q_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, q_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            # The attention mask is for padding, shape [bsz, 1, q_len, kv_seq_len]
             if attention_mask.size() != (bsz, 1, q_len, q_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, q_len)}, but is {attention_mask.size()}"
                )
            # Add the mask (typically -inf for positions to ignore)
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states) # [bsz, num_heads, q_len, head_dim]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
             raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous() # [bsz, q_len, num_heads, head_dim]
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim) # [bsz, q_len, hidden_dim]

        attn_output = self.o_proj(attn_output)
        return attn_output


class LLaDATransformerBlock(nn.Module):
    # A single block of the LLaDA Transformer (Attention + FFN)
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.attention = LLaDAAttention(config)
        self.ffn = FFN(config)
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        # Dropout can be added here if needed (config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # --- Pre-RMSNorm structure (common in Llama) ---
        residual = hidden_states
        normalized_hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.attention(
            normalized_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # Residual connection
        hidden_states = residual + attn_output # Apply residual before FFN norm

        # Fully Connected
        residual = hidden_states
        normalized_hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.ffn(normalized_hidden_states)
        # Residual connection
        hidden_states = residual + ffn_output

        return hidden_states

# --- Main Model ---

class LLaDAModel(PreTrainedModel):
    # The core LLaDA model architecture
    config_class = LLaDAConfig

    def __init__(self, config: LLaDAConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id # Usually 0 or specific padding ID
        self.vocab_size = config.vocab_size
        self.mask_token_id = config.mask_token_id

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim, self.padding_idx)
        self.layers = nn.ModuleList([LLaDATransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps) # Final layer norm

        # LM Head (predicts logits over vocab) - Shares weights with embedding optional
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # LLaDA doesn't use a causal mask, but needs a padding mask.
        # This function is adapted from Llama's _prepare_decoder_attention_mask
        # but ensures no causal component is added.
        # Create attention mask for padding
        # attention_mask: [bsz, seq_len] -> [bsz, 1, q_len, kv_seq_len]
        if attention_mask is not None:
             # Create the attention mask for padding
             # Expand mask from [bsz, seq_len] to [bsz, 1, q_len, kv_seq_len]
             # For self-attention, q_len == kv_seq_len == seq_len
             expanded_attn_mask = attention_mask[:, None, None, :].expand(input_shape[0], 1, input_shape[1], input_shape[1]).to(inputs_embeds.dtype)
             inverted_mask = 1.0 - expanded_attn_mask
             # Masked positions get filled with a large negative value (-inf like)
             masked_value = torch.finfo(inputs_embeds.dtype).min
             # Final mask: 0.0 for valid tokens, large negative for padding
             final_attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), masked_value)
             return final_attention_mask
        else:
            return None # No padding mask needed if attention_mask is None

    def forward(
        self,
        input_ids: torch.LongTensor, # Represents xt (partially masked sequence)
        attention_mask: torch.Tensor | None = None, # Mask for padding tokens (1 = keep, 0 = mask out)
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None, # Represents x0 (original sequence)
        return_dict: bool | None = None,
        output_attentions: bool | None = None, # Not implemented here
        output_hidden_states: bool | None = None, # Not implemented here
    ) -> tuple | dict: # Adjust return type based on Hugging Face conventions

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Embedding
        inputs_embeds = self.embedding(input_ids)
        batch_size, seq_length, _ = inputs_embeds.shape

        # 2. Prepare Attention Mask (for padding) & Position IDs
        if position_ids is None:
             position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device)
             position_ids = position_ids.unsqueeze(0).view(-1, seq_length) # [1, seq_len] -> broadcast later

        # LLaDA does NOT use causal mask. Create padding mask if needed.
        # HF models expect attention_mask = 1 for tokens to *attend* to, 0 for padding
        # MHA expects mask = True for tokens to *ignore*. We'll adapt inside LLaDAAttention
        # The `_prepare_decoder_attention_mask` adapted function prepares a mask suitable for adding to attention scores.
        attn_mask_for_addition = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, 0 # 0 for past_key_values_length
        )


        # 3. Transformer Blocks
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attn_mask_for_addition,
                position_ids=position_ids,
            )

        # 4. Final Normalization
        hidden_states = self.norm(hidden_states)

        # 5. LM Head (Logit Prediction)
        logits = self.lm_head(hidden_states) # [batch_size, seq_len, vocab_size]

        # 6. Loss Calculation (LLaDA Specific)
        loss = None
        if labels is not None:
            # Identify masked positions in the input
            # Make sure mask_token_id is correctly set in the config!
            masked_positions = (input_ids == self.mask_token_id) # [batch_size, seq_len] boolean tensor

            if masked_positions.sum() == 0:
                 # Handle case with no masked tokens gracefully (e.g., during eval or specific sampling steps)
                 # Set loss to 0 or handle as needed
                 print("Warning: No masked tokens found in batch, loss calculation skipped.")
                 # Or set loss based on some default/average if appropriate for your training loop
                 loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

            else:
                # Select logits and labels only at masked positions
                masked_logits = logits[masked_positions] # Shape: [num_masked_tokens, vocab_size]
                masked_labels = labels[masked_positions] # Shape: [num_masked_tokens]

                # Compute CrossEntropyLoss (expects logits, not probabilities)
                # The loss is averaged over the masked tokens in the batch
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(masked_logits, masked_labels)


        if not return_dict:
            output = (logits,) + (None,) * (2 if output_hidden_states or output_attentions else 0) # Placeholder for hidden_states/attentions
            return ((loss,) + output) if loss is not None else output

        # Using a simple dictionary return for clarity, can adapt to HF's ModelOutput later
        return {
            "loss": loss,
            "logits": logits,
            # "hidden_states": hidden_states if output_hidden_states else None, # Add if needed
            # "attentions": None, # Add if needed
        }


# --- Helper Functions & Main Execution ---

def load_config_yaml(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def mask_sequence(input_ids, mask_token_id, mask_ratio=0.15, ignore_token_id=-100):
    """
    Applies random masking to a sequence (batch).
    Args:
        input_ids (torch.Tensor): Batch of input token IDs [batch_size, seq_len].
        mask_token_id (int): The ID of the [MASK] token.
        mask_ratio (float): Probability of masking each token.
        ignore_token_id (int): ID for tokens that should not be masked (e.g., padding, BOS, EOS).
    Returns:
        masked_input_ids (torch.Tensor): Input sequence with tokens randomly replaced by mask_token_id.
        original_input_ids (torch.Tensor): A copy of the original input_ids to be used as labels.
    """
    original_input_ids = input_ids.clone()
    masked_input_ids = input_ids.clone()

    # Determine which tokens are eligible for masking (not special tokens)
    can_mask = (input_ids != ignore_token_id) # Add other special tokens if needed (e.g., BOS, EOS)
    if input_ids.dim() > 1 and input_ids.shape[0] > 0: # Handle padding if ignore_token_id is pad_token_id
        can_mask &= (input_ids != input_ids[0,0]) # Simple check assuming BOS/CLS is first token
        can_mask &= (input_ids != input_ids[0,-1]) # Simple check assuming EOS/SEP is last token

    # Create random mask based on probability
    probability_matrix = torch.full(input_ids.shape, mask_ratio, device=input_ids.device)
    masked_indices = torch.bernoulli(probability_matrix).bool() & can_mask

    # Apply the mask
    masked_input_ids[masked_indices] = mask_token_id

    return masked_input_ids, original_input_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Allow overriding config values via command line
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--intermediate_size', type=int)
    parser.add_argument('--num_hidden_layers', type=int)
    parser.add_argument('--num_attention_heads', type=int)
    parser.add_argument('--mask_token_id', type=int, required=True, help="ID of the mask token is required.") # Make MASK ID required
    parser.add_argument('--mask_ratio', type=float, default=0.5, help="Default mask ratio for demo.") # Example default
    # Add other config args as needed

    args = parser.parse_args()

    # --- Configuration Setup ---
    # Start with default config
    config = LLaDAConfig()

    # Override with args if provided
    config_overrides = {k: v for k, v in vars(args).items() if v is not None and hasattr(config, k)}
    config.__dict__.update(config_overrides)
    print("Using Configuration:")
    print(config)

    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # --- Model Initialization ---
    model = LLaDAModel(config).to(device)
    print(f"\nModel Initialized. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Dummy Data and Masking Example ---
    print("\n--- Running Dummy Forward Pass ---")
    batch_size = 2
    seq_len = 64
    # Create dummy input IDs (replace with real tokenized data)
    input_ids_original = torch.randint(low=10, high=config.vocab_size - 1, size=(batch_size, seq_len), device=device)

    # Apply masking (using the configured mask_token_id and provided ratio)
    mask_ratio_t = args.mask_ratio # Example: Use the command-line arg or a fixed value
    input_ids_masked, labels_original = mask_sequence(
        input_ids_original,
        config.mask_token_id,
        mask_ratio=mask_ratio_t
    )

    # Create a simple attention mask (assuming no padding here, all tokens are valid)
    # If you have padding, create a mask where 0 indicates padding.
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

    print(f"Original Input IDs (sample): {input_ids_original[0, :15]}...")
    print(f"Masked Input IDs   (sample): {input_ids_masked[0, :15]}...")
    print(f"Labels           (sample): {labels_original[0, :15]}...") # Should match Original
    print(f"Number of masked tokens in batch: {(input_ids_masked == config.mask_token_id).sum().item()}")


    # --- Forward Pass ---
    with torch.no_grad(): # Use no_grad for inference/demonstration
      outputs = model(
          input_ids=input_ids_masked,
          attention_mask=attention_mask,
          labels=labels_original, # Provide original sequence as labels
          return_dict=True
      )

    # --- Output Inspection ---
    print(f"\nOutput Logits Shape: {outputs['logits'].shape}") # Expected: [batch_size, seq_len, vocab_size]
    if outputs['loss'] is not None:
        print(f"Calculated Loss: {outputs['loss'].item():.4f}")
    else:
        print("Loss was not calculated (e.g., no masked tokens).")

    # Example: Check prediction for the first masked token in the first batch instance
    first_batch_mask_indices = (input_ids_masked[0] == config.mask_token_id).nonzero(as_tuple=True)[0]
    if len(first_batch_mask_indices) > 0:
        first_masked_idx = first_batch_mask_indices[0].item()
        predicted_token_id = torch.argmax(outputs['logits'][0, first_masked_idx]).item()
        actual_token_id = labels_original[0, first_masked_idx].item()
        print(f"\nExample Prediction @ First Masked Position (idx {first_masked_idx}):")
        print(f"  Predicted Token ID: {predicted_token_id}")
        print(f"  Actual Token ID:    {actual_token_id}")