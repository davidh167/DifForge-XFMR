import torch
import torch.nn as nn
from transformers import LongT5ForConditionalGeneration, AutoTokenizer

# ------------------------------
# Structure Decoder: Global latent to 32 layer latents.
# ------------------------------
class StructureDecoder(nn.Module):
    def __init__(self, global_latent_dim, layer_embedding_dim, num_layers=32):
        super().__init__()
        self.num_layers = num_layers
        self.fc = nn.Linear(global_latent_dim, num_layers * layer_embedding_dim)
    
    def forward(self, global_latent):
        # global_latent: (batch_size, global_latent_dim) 
        if global_latent.dim() == 1:
            global_latent = global_latent.unsqueeze(0)
        out = self.fc(global_latent)  # (batch_size, num_layers * layer_embedding_dim)
        layer_latents = out.view(global_latent.size(0), self.num_layers, -1)  # (batch_size, num_layers, layer_embedding_dim)
        return layer_latents

# ------------------------------
# Layer Decoder with LED for Conditional Generation.
# ------------------------------
class LayerDecoderLongT5(nn.Module):
    """
    This module decodes a single layer latent into a token sequence.
    It uses a pretrained LongT5ForConditionalGeneration as the decoder.
    A small projection converts the layer latent into a learned prefix.
    """
    def __init__(self, layer_embedding_dim, prompt_length, vocab_size, pretrained_model_name="allenai/LED-base-16384", max_length=1024):
        super().__init__()
        self.prompt_length = prompt_length
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Load a pretrained LED model for conditional generation.
        self.longt5 = LongT5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.longt5.gradient_checkpointing_enable()
        self.longt5.resize_token_embeddings(vocab_size)
        print("LED vocab size: " ,self.led.config.vocab_size)

        # Projection: map layer latent to a sequence of prompt embeddings.
        self.decoder_hidden_size = self.longt5.config.d_model  # e.g., 1024 for LED-base-16384.

        self.prefix_proj = nn.Linear(layer_embedding_dim, prompt_length * self.decoder_hidden_size)

    def forward(self, layer_latent, input_ids=None, attention_mask=None):
        """
        If input_ids is provided, use teacher forcing; otherwise, use generate() for inference.
        Args:
            layer_latent: (batch_size, layer_embedding_dim)
            input_ids: (batch_size, seq_length) – target tokens for teacher forcing.
            attention_mask: (batch_size, seq_length)
        Returns:
            logits of shape (batch_size, seq_length, vocab_size) in teacher forcing mode.
            In inference mode, returns the generate() output.
        """
        batch_size = layer_latent.size(0)

        print("batch size init: ", batch_size)

        # Project to prefix embeddings.
        prefix_embeddings = self.prefix_proj(layer_latent)  # (batch_size, prompt_length * decoder_hidden_size)
        prefix_embeddings = prefix_embeddings.view(batch_size, self.prompt_length, self.decoder_hidden_size)
        
        if input_ids is not None:
            # Teacher forcing: get embeddings for target sequence
            target_embeddings = self.longt5.get_input_embeddings()(input_ids)  # (batch_size, seq_length, decoder_hidden_size)
            # Concatenate prefix embeddings to the target embeddings
            inputs_embeds = torch.cat([prefix_embeddings, target_embeddings], dim=1)  # (batch_size, prompt_length+seq_length, decoder_hidden_size)
            # Build attention mask: ones for prefix and then use the provided mask for target tokens.
            prefix_mask = torch.ones(batch_size, self.prompt_length, device=layer_latent.device)
            full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1) if attention_mask is not None else None
            outputs = self.longt5(
                decoder_inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            # Slice off the prefix part to get logits corresponding to the target tokens.
            logits = outputs.logits[:, self.prompt_length:, :]  # (batch_size, seq_length, vocab_size)
            return logits
        else:
            # In inference mode, use the prefix as a prompt.
            total_length = self.prompt_length + self.max_length
            outputs = self.longt5.generate(
                decoder_inputs_embeds=prefix_embeddings,
                max_length=total_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            return outputs

# ------------------------------
# Hierarchical Decoder: Combines StructureDecoder and LayerDecoderLongT5.
# ------------------------------
class HierarchicalDecoder(nn.Module):
    """
    The hierarchical decoder first uses a structure decoder to expand a global latent into
    per-layer latent vectors, then decodes each layer latent into a token sequence using LongT5.
    """
    def __init__(self, global_latent_dim, layer_embedding_dim, vocab_size, prompt_length, num_layers=32, max_length=1024, pretrained_model_name="allenai/LED-base-16384"):
        super().__init__()
        self.structure_decoder = StructureDecoder(global_latent_dim, layer_embedding_dim, num_layers)
        self.layer_decoder = LayerDecoderLongT5(layer_embedding_dim, prompt_length, vocab_size, pretrained_model_name, max_length)
        self.num_layers = num_layers
    
    def forward(self, global_latent, decoder_input_ids=None, attention_mask=None):
        # global_latent: (batch_size, global_latent_dim)
        layer_latents = self.structure_decoder(global_latent)  # (batch_size, num_layers, layer_embedding_dim)
        decoded_layers = []
        # For each layer, decode using the LED-based decoder.
        for i in range(self.num_layers):
            current_latent = layer_latents[:, i, :]  # (batch_size, layer_embedding_dim)
            layer_logits = self.layer_decoder(current_latent, 
                                              input_ids=decoder_input_ids,
                                              attention_mask=attention_mask
                                              )
            decoded_layers.append(layer_logits)
        # Stack the outputs: (batch_size, num_layers, seq_length, vocab_size)
        full_structure_logits = torch.stack(decoded_layers, dim=1)
        return full_structure_logits