import torch
import torch.nn as nn
from transformers import LEDForConditionalGeneration, LEDTokenizer

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
class LayerDecoderLED(nn.Module):
    """
    Uses a pretrained LEDForConditionalGeneration as the base decoder.
    The layer latent is projected into a prefix (of fixed length) that conditions the LED decoder.
    """
    def __init__(self, layer_embedding_dim, prompt_length, vocab_size, pretrained_model_name="allenai/LED-base-16384", max_length=1024):
        super().__init__()
        self.prompt_length = prompt_length
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Load a pretrained LED model for conditional generation.
        self.led = LEDForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.led.gradient_checkpointing_enable()
        self.led.resize_token_embeddings(vocab_size)
        print("LED vocab size: " ,self.led.config.vocab_size)

        # Projection: map layer latent to a sequence of prompt embeddings.
        decoder_hidden_size = self.led.config.d_model  # e.g., 1024 for LED-base-16384.
        self.decoder_hidden_size = decoder_hidden_size

        self.prefix_proj = nn.Linear(layer_embedding_dim, prompt_length * decoder_hidden_size)

    def forward(self, layer_latent, input_ids=None, attention_mask=None):
        """
        layer_latent: (batch_size, layer_embedding_dim)
        If input_ids is provided, perform teacher forcing.
        Otherwise, use generate() for inference.
        """
        batch_size = layer_latent.size(0)

        print("batch size init: ", batch_size)

        # Project to prefix embeddings.
        prefix_embeddings = self.prefix_proj(layer_latent)  # (batch_size, prompt_length * decoder_hidden_size)
        prefix_embeddings = prefix_embeddings.view(batch_size, self.prompt_length, self.decoder_hidden_size)
        
        if input_ids is not None:
            # Teacher-forcing mode: obtain embeddings for the target sequence.
            # Ensure input_ids is 2D (batch_size, seq_length)
            if input_ids.dim() > 2:
                input_ids = input_ids.squeeze(1)
            decoder_input_embeds = self.led.get_input_embeddings()(input_ids) # (batch_size, seq_length, decoder_hidden_size)
            print("decoder_input_embeds", decoder_input_embeds.size())
            print("prefix_embeddings", prefix_embeddings.size())
            
            # Truncate decoder input embeddings to 1014 tokens
            decoder_input_embeds = decoder_input_embeds[:, :1014, :]
            print("truncated decoder_input_embeds ", decoder_input_embeds.size())

            # Prepend the prefix embeddings to the target embeddings.
            inputs_embeds = torch.cat([prefix_embeddings, decoder_input_embeds], dim=1)  # (batch_size, prompt_length+seq_length, decoder_hidden_size)
            print("inputs_embeds", inputs_embeds.size()) # Expected: [1, 1024, 768]

            # Build a 2D attention mask.
            # Assume prefix_embeddings: [batch_size, 10, hidden_size] and decoder_input_embeds: [batch_size, 1014, hidden_size]
            prefix_mask = torch.ones(prefix_embeddings.shape[0], prefix_embeddings.shape[1],
                                    dtype=torch.long, device=layer_latent.device)
            target_mask = torch.ones(decoder_input_embeds.shape[0], decoder_input_embeds.shape[1],
                                    dtype=torch.long, device=layer_latent.device)
            # Concatenate: expected shape [batch_size, 10+1014] = [1, 1024]
            full_attention_mask = torch.cat([prefix_mask, target_mask], dim=1)
            print("Full attention mask shape:", full_attention_mask.shape)  # Should be [1, 1024]

            # Create dummy encoder outputs for LED.
            batch_size = inputs_embeds.size(0)
            
            # Create a dummy encoder output to satisfy LED's requirements.
            # LED expects encoder_outputs to be a tuple where the first element is a tensor of shape (batch_size, encoder_seq_length, d_model).
            dummy_encoder_outputs = (
                torch.zeros(batch_size, 1, self.led.config.d_model, device=inputs_embeds.device),
                )

            print("LED d_model:", self.led.config.d_model)
            print("LED max_position_embeddings:", self.led.config.max_decoder_position_embeddings)
            print("Concatenated sequence length:", inputs_embeds.size(1))

            outputs = self.led(
                # inputs_embeds=inputs_embeds, 
                encoder_outputs = dummy_encoder_outputs,
                decoder_inputs_embeds=inputs_embeds,
                decoder_attention_mask=full_attention_mask, 
                # decoder_input_ids = None, 
                return_dict=True
                )
            
            print("LED output logits shape:", outputs.logits.shape)
            # We only want the logits for the target part (i.e. after the prefix).
            logits = outputs.logits[:, self.prompt_length:, :]  # (batch_size, seq_length, vocab_size)
            print("Logits after slicing prefix:", logits.shape)
            return logits
        else:
            total_length = self.prompt_length + 1014  # desired total length
            outputs = self.led.generate(
                decoder_inputs_embeds=prefix_embeddings,
                max_length=total_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            # In inference mode, use the prefix as a prompt.
            # outputs = self.led.generate(decoder_inputs_embeds=prefix_embeddings, 
            #                             max_length=self.prompt_length + self.max_length,
                                        
                                        # )
            print("Generating...")
            return outputs

# ------------------------------
# Hierarchical Decoder: Combines StructureDecoder and LayerDecoderLED.
# ------------------------------
class HierarchicalDecoder(nn.Module):
    def __init__(self, global_latent_dim, layer_embedding_dim, vocab_size, prompt_length, num_layers=32, max_length=1024, pretrained_model_name="allenai/LED-base-16384"):
        super().__init__()
        self.structure_decoder = StructureDecoder(global_latent_dim, layer_embedding_dim, num_layers)
        self.layer_decoder = LayerDecoderLED(layer_embedding_dim, prompt_length, vocab_size, pretrained_model_name, max_length)
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