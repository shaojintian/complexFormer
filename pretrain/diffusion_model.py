import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# import torchviz # Optional for visualization
# from torch.utils.tensorboard import SummaryWriter # Optional for logging
import yaml
import argparse
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
import logging
import hydra
from omegaconf import DictConfig, OmegaConf # Import OmegaConf for Hydra config access
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, Any, Tuple, Union
# from rotary_embedding_torch import RotaryEmbedding # Removed for standard diffusion setup
import math # Needed for sinusoidal embeddings
from rich import traceback
traceback.install(show_locals=True)

logger: logging.Logger = logging.getLogger(__name__)
logger.propagate = False
# Clear existing handlers if necessary
if logger.hasHandlers():
    logger.handlers.clear()
# Create logs directory if it doesn't exist
import os
os.makedirs("logs", exist_ok=True)
logger.addHandler(logging.FileHandler("logs/diffusion_model.log")) # Changed log file name
logger.setLevel(logging.INFO) # Set level appropriately

# --- Configuration ---
class CustomConfig(PretrainedConfig):
    model_type: str = "diffusion_transformer" # Changed model type name

    def __init__(self,
                 vocab_size: int = 30522,
                 hidden_dim: int = 512,
                 intermediate_size: int = 1024,
                 max_seq_len: int = 64, # Diffusion often uses shorter sequences
                 n_layers: int = 8,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1, # Slightly higher dropout might be beneficial
                 time_embed_dim: Optional[int] = None, # Dimension for timestep embedding
                 # --- Diffusion Specific Params (Example) ---
                 num_diffusion_timesteps: int = 1000,
                 beta_schedule: str = "linear", # or cosine, etc.
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 # --- End Diffusion Params ---
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.vocab_size: int = vocab_size
        self.hidden_dim: int = hidden_dim
        self.intermediate_size: int = intermediate_size
        self.max_seq_len: int = max_seq_len
        self.n_layers: int = n_layers
        self.num_attention_heads: int = num_attention_heads
        self.dropout: float = dropout
        # If time_embed_dim not specified, default to hidden_dim
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_dim
        self.debug : bool = kwargs.get("debug", False)

        # Add diffusion params to config object
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

# --- Modules ---

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Creates sinusoidal timestep embeddings.
    """
    def __init__(self, dim: int, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, time: Tensor) -> Tensor:
        # Input time: (batch_size,) tensor of integers
        device = time.device
        half_dim = self.dim // 2
        # exponents: (half_dim,)
        exponents = torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
        # freqs: (half_dim,)
        freqs = torch.exp(-math.log(self.max_period) * exponents)
        # args: (batch_size, half_dim)
        args = time[:, None].float() * freqs[None, :]
        # embeddings: (batch_size, dim)
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # If dim is odd, pad with zero
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings

class FFN(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        # Use GeGLU variant for potentially better performance
        self.w1 = nn.Linear(config.hidden_dim, config.intermediate_size)
        self.w3 = nn.Linear(config.hidden_dim, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_dim)
        self.activation_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_gelu = self.activation_fn(self.w1(x))
        x_linear = self.w3(x)
        x = x_gelu * x_linear
        x = self.dropout(x)
        x = self.w2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6): # Increased eps slightly
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 for stability
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(dtype) # Cast back
        return self.weight * x

class TransformerBlock(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
        self.ffn: FFN = FFN(config)
        self.norm1: RMSNorm = RMSNorm(config.hidden_dim)
        self.norm2: RMSNorm = RMSNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        # No causal mask here, let MHA handle it if needed (usually not for noise pred)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        # padding_mask: (batch_size, seq_len), True where padded
        # Note: For standard diffusion noise prediction, causal mask is often NOT used
        # unless you are doing some form of autoregressive diffusion.
        batch_size, seq_len, _ = x.shape
        logger.debug(f"TransformerBlock Input shape: {x.shape}")

        residual: Tensor = x
        x_norm = self.norm1(x)

        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=padding_mask, # Use the provided padding mask (True means ignore)
            attn_mask=None # No causal mask for standard noise prediction
        )
        x = residual + self.dropout(attn_output) # Apply dropout after attention

        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output) # Apply dropout after FFN

        logger.debug(f"TransformerBlock Output shape: {x.shape}")
        return x

# --- Main Diffusion Model ---

class DiffusionModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.config: CustomConfig = config

        # Token embeddings (still needed for initial/final mapping during training/sampling)
        self.embedding: nn.Embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Positional embeddings (standard absolute positional embeddings)
        self.position_embedding: nn.Embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Timestep embedding projection MLP
        time_embed_dim = config.time_embed_dim
        self.time_embedding: nn.Sequential = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, config.hidden_dim) # Project to hidden_dim
        )

        # Input projection (optional: map embeddings before adding time/pos)
        # self.input_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Transformer Blocks
        self.transformer_blocks: nn.ModuleList = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final normalization and output projection to predict noise
        self.final_norm: RMSNorm = RMSNorm(config.hidden_dim)
        self.output_projection: nn.Linear = nn.Linear(config.hidden_dim, config.hidden_dim) # Predict noise in embedding space

        self.gradient_checkpointing = False
        self.debug = config.debug
        self._toggle_logger()

        # Initialize weights
        self.post_init()

    def _toggle_logger(self):
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    def get_input_embeddings(self):
        """ Helper to access token embeddings """
        return self.embedding

    def get_output_embeddings(self):
        """ Helper to access the final projection layer (if needed for tying weights)"""
        # Usually not tied in diffusion, but good practice
        return self.output_projection

    def forward(
        self,
        noisy_embeddings: Tensor,           # Input is now continuous noisy embeddings
        timesteps: Tensor,                  # Input timesteps
        attention_mask: Optional[Tensor] = None, # Standard attention mask (True for non-padded)
        conditioning_embeddings: Optional[Tensor] = None, # Optional conditioning
        return_dict: Optional[bool] = None, # For HF compatibility (optional)
    ) -> Union[Tensor, Tuple]: # Output is predicted noise
        # noisy_embeddings: (batch_size, seq_len, hidden_dim)
        # timesteps: (batch_size,) tensor of integers
        # attention_mask: (batch_size, seq_len) bool/int tensor (True or 1 for non-masked)
        # conditioning_embeddings: (batch_size, cond_seq_len, hidden_dim) (optional)

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict # For HF Trainer

        if torch.isnan(noisy_embeddings).any() or torch.isinf(noisy_embeddings).any():
            raise ValueError("noisy_embeddings Tensor contains NaN or Inf values.")
        if torch.isnan(timesteps).any() or torch.isinf(timesteps).any():
             raise ValueError("timesteps Tensor contains NaN or Inf values.")

        batch_size, seq_len, hidden_dim = noisy_embeddings.shape
        device = noisy_embeddings.device
        logger.debug(f"Input noisy_embeddings shape: {noisy_embeddings.shape}")
        logger.debug(f"Input timesteps shape: {timesteps.shape}")
        logger.debug(f"Input attention_mask shape: {attention_mask.shape if attention_mask is not None else 'None'}")

        # 1. Prepare attention mask for MHA (needs True for PADDED tokens)
        if attention_mask is None:
            mha_padding_mask = None # Assume all tokens are valid
        else:
            # Ensure mask is boolean and invert it
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            # Ensure attention_mask has shape (batch_size, seq_len)
            if attention_mask.dim() == 1: # If mask is just seq_len, expand
                 attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1)
            elif attention_mask.dim() > 2: # Handle extra dims if present (e.g., from broadcasting)
                 attention_mask = attention_mask.squeeze() # Simple squeeze, might need adjustment
                 if attention_mask.dim() != 2:
                      raise ValueError(f"Attention mask has unexpected shape {attention_mask.shape}")

            mha_padding_mask = ~attention_mask # MHA needs True for PAD tokens
            logger.debug(f"MHA padding mask shape: {mha_padding_mask.shape}")

        # 2. Get positional embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0) # Shape: (1, seq_len)
        position_embeddings = self.position_embedding(position_ids) # Shape: (1, seq_len, hidden_dim)
        logger.debug(f"Position Embeddings shape: {position_embeddings.shape}")

        # 3. Get timestep embeddings
        time_emb = self.time_embedding(timesteps) # Shape: (batch_size, hidden_dim)
        # Expand time embedding to match sequence length
        time_emb = time_emb.unsqueeze(1) # Shape: (batch_size, 1, hidden_dim)
        # No need to expand to seq_len here, just add broadcasted
        logger.debug(f"Time Embeddings shape (unsqueezed): {time_emb.shape}")

        # 4. Combine inputs
        # x = self.input_proj(noisy_embeddings) # Optional projection
        x = noisy_embeddings + position_embeddings + time_emb
        logger.debug(f"Combined Input shape: {x.shape}")

        # --- Optional: Add Conditioning ---
        if conditioning_embeddings is not None:
            # TODO: Implement proper conditioning (e.g., cross-attention, FiLM layers)
            # Simple addition example (assumes conditioning is projected & compatible)
            logger.warning("Conditioning mechanism not fully implemented (using simple addition placeholder)")
            # If global conditioning:
            # cond_emb = conditioning_embeddings.unsqueeze(1) # (batch, 1, hidden_dim)
            # x = x + cond_emb
            pass

        # 5. Pass through Transformer Blocks
        for i, block in enumerate(self.transformer_blocks):
            layer_name = f"layer_{i}"
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(inputs[0], padding_mask=inputs[1])
                    return custom_forward

                x = checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    mha_padding_mask,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
                logger.debug(f"Transformer block {i} output shape (ckpt): {x.shape}")
            else:
                x = block(x, padding_mask=mha_padding_mask)
                logger.debug(f"Transformer block {i} output shape (no ckpt): {x.shape}")

        # 6. Final normalization and projection
        x = self.final_norm(x)
        predicted_noise = self.output_projection(x) # Shape: (batch_size, seq_len, hidden_dim)
        logger.debug(f"Final Norm output shape: {x.shape}")
        logger.info(f"Predicted Noise output shape: {predicted_noise.shape}")

        # For HF Trainer compatibility
        # if not return_dict:
        #     return (predicted_noise,)
        # return BaseModelOutputWithPooling(...) # Or a custom output class

        return predicted_noise # Return the predicted noise


# --- Loading Config --- (Keep your load_config function)
def load_config(config_path: str) -> CustomConfig:
    with open(config_path, 'r') as f:
        config_dict: Dict[str, Any] = yaml.safe_load(f)
    # Add necessary diffusion parameters if missing from yaml for backward compat.
    defaults = {
        'num_diffusion_timesteps': 1000,
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'time_embed_dim': None # Let constructor handle default
    }
    for k, v in defaults.items():
        if k not in config_dict:
            logger.warning(f"'{k}' not found in config yaml, using default value: {v}")
            config_dict[k] = v
    return CustomConfig(**config_dict)


# --- Testing Function ---
def test_model(model: DiffusionModel, config: CustomConfig) -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Testing model on device: {device}")
    model.to(device)
    model.eval() # Set model to evaluation mode for testing

    # --- Generate Dummy Diffusion Inputs ---
    batch_size = 4 # Use a smaller batch size for testing
    seq_len = config.max_seq_len
    hidden_dim = config.hidden_dim
    vocab_size = config.vocab_size

    # 1. Dummy initial embeddings (e.g., from random tokens)
    # In a real scenario, this comes from embedding input_ids
    # input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # initial_embeddings = model.get_input_embeddings()(input_ids) # (batch, seq_len, hidden_dim)
    # For testing the forward pass, we can just start with random embeddings
    initial_embeddings = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # 2. Dummy Timesteps
    # Sample random timesteps from [0, num_diffusion_timesteps - 1]
    timesteps = torch.randint(0, config.num_diffusion_timesteps, (batch_size,), device=device).long()

    # 3. Dummy Noise
    # This is the noise we *expect* the model to predict
    noise = torch.randn_like(initial_embeddings)

    # 4. Create Noisy Embeddings (Simplified Forward Process)
    # In a real training loop, you'd use the alpha/beta schedule to add noise correctly
    # noisy_embeddings = sqrt(alpha_cumprod_t) * initial_embeddings + sqrt(1 - alpha_cumprod_t) * noise
    # For just testing the model's forward pass shape, random noise suffices as input
    noisy_embeddings = torch.randn_like(initial_embeddings) # Use random input for shape check
    logger.info(f"Test noisy_embeddings shape: {noisy_embeddings.shape}")
    logger.info(f"Test timesteps shape: {timesteps.shape}")

    # 5. Dummy Attention Mask (e.g., some padding)
    # True = non-padded, False = padded
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    # Example: Mask out the last 10 tokens for the first batch item
    if seq_len > 10:
        attention_mask[0, -10:] = False
    logger.info(f"Test attention_mask shape: {attention_mask.shape}")


    # --- Model Forward Pass ---
    with torch.no_grad(): # No need to compute gradients for testing
        predicted_noise: Tensor = model(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask,
            # conditioning_embeddings=None, # Add if testing conditioning
        )

    logger.info(f"Model predicted_noise shape: {predicted_noise.shape}")

    # --- Assertions ---
    expected_shape = (batch_size, seq_len, hidden_dim)
    assert predicted_noise.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, Got {predicted_noise.shape}"
    assert predicted_noise.device == device, "Output tensor is on the wrong device."

    # --- Calculate Dummy Loss (MSE) ---
    # We compare the predicted noise with the actual noise added
    loss_fn = nn.MSELoss()
    loss = loss_fn(predicted_noise, noise) # Compare model output with the target noise
    logger.info(f"Test MSE Loss: {loss.item():.4f}") # Log the scalar loss value

    logger.info("Model test passed successfully!")


# --- Main Execution ---
@hydra.main(config_path='.', config_name="config.yaml", version_base=None) # Added version_base
def main(cfg: DictConfig) -> None: # Use DictConfig type hint from OmegaConf
    # print(OmegaConf.to_yaml(cfg)) # Useful for debugging hydra config

    # --- Setup ---
    # Use Hydra's config directly if possible, otherwise load from file specified in args/config
    # This example prioritizes Hydra's config structure (`cfg.model`)
    try:
        # Assuming config.yaml has a structure like:
        # model:
        #   save_dir: ...
        #   <parameters for CustomConfig> ...
        # training:
        #   batch_size: ...
        # Ensure parameters match CustomConfig or are nested under a sub-key
        model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
        # Make sure vocab_size etc. are directly under model or handle nesting
        config = CustomConfig(**model_config_dict)

        # Get other parameters if needed
        batch_size_from_cfg = cfg.training.batch_size # Example access
        save_dir = cfg.model.save_dir

    except Exception as e:
        logger.error(f"Error processing Hydra config: {e}")
        logger.info("Falling back to loading config via --config argument (if provided).")
        # Fallback to argparse (less flexible with Hydra)
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the model config file (YAML)')
        # Parse only known args if other hydra args exist
        args, unknown = parser.parse_known_args()
        config = load_config(args.config) # Load config using the YAML file
        # Need to manually get other params like batch_size, save_dir if not in model YAML
        # This part becomes less integrated with Hydra
        batch_size_from_cfg = 4 # Placeholder default
        save_dir = "diffusion_model_save" # Placeholder default
        logger.warning("Hydra config structure mismatch or error. Loaded config from file, but other parameters might use defaults.")


    # Register model type with Auto* classes
    AutoConfig.register(config.model_type, CustomConfig) # Use model_type from config
    AutoModel.register(CustomConfig, DiffusionModel)

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Model Initialization ---
    model: DiffusionModel = DiffusionModel(config=config)
    model.to(device)
    logger.info(f"Model Class: {model.__class__.__name__}")
    logger.info(f"Model Config: {config.to_dict()}") # Log the configuration used

    # --- Test Model ---
    # Pass a temporary config object to test_model, potentially with overrides
    test_cfg_dict = config.to_dict()
    # test_cfg_dict['max_seq_len'] = 32 # Can override for testing if needed
    test_config_obj = CustomConfig(**test_cfg_dict)
    test_model(model, test_config_obj) # Use the config object for testing

    # --- Gradient Checkpointing (Optional) ---
    if cfg.get('training', {}).get('gradient_checkpointing', False): # Check Hydra config safely
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        # Verify it's enabled
        if hasattr(model, 'gradient_checkpointing') and model.gradient_checkpointing:
             logger.info("Gradient checkpointing successfully enabled.")
        else:
             logger.warning("Attempted to enable gradient checkpointing, but flag is not set.")


    # --- Calculate Parameters ---
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {num_params / 1e6:.2f}M")
    logger.info(f"Trainable model parameters: {num_trainable_params / 1e6:.2f}M")

    # --- Save Model ---
    try:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        logger.info(f"Model configuration and weights saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving model to {save_dir}: {e}")


if __name__ == '__main__':
    # Setup traceback install here as well if running script directly
    traceback.install(show_locals=True)
    main()