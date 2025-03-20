"""
Author: David Hernandez

This file serves to define the StructureAggregator model that combines 32 layer embeddings into a global structure representation.

The StructureAggregator model is a simple TransformerEncoder that averages the 32 layer embeddings to produce a global structure representation.
The model is used to train a LongT5 model to generate 32-layer structures.

"""

import torch
import torch.nn as nn
import json, ast
import numpy as np
from transformers import LongformerSelfAttention

# ------------------------------
# 1. Load the Pretrained Layer Encoder (using LongT5)
# ------------------------------
from transformers import LongT5ForConditionalGeneration, LongT5Tokenizer

def load_layer_encoder(model_path, tokenizer_path):
    """
    Loads a LongT5 model and tokenizer for encoding a layer.
    (For our purposes we use the conditional generation model and use its encoder.)
    """
    tokenizer = LongT5Tokenizer.from_pretrained(tokenizer_path)
    model = LongT5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

# The function to encode a single layer remains similar;
# here we extract the encoder's last hidden state for the first token.
def encode_layer(layer_tokens, layer_encoder, tokenizer, device):
    inputs = tokenizer(layer_tokens, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Forward through the model's encoder (LongT5ForConditionalGeneration's encoder is available via model.encoder)
    with torch.no_grad():
        encoder_outputs = layer_encoder.encoder(**inputs, output_hidden_states=True)
    # Use the last hidden state and take the representation of the first token as a summary.
    hidden_states = encoder_outputs.last_hidden_state  # Shape: (batch_size, seq_length, d_model)
    # layer_embedding = hidden_states[:, 0, :]  # (batch_size, d_model)
    layer_embedding = hidden_states.mean(dim=1)  # Take the mean across all tokens
    return layer_embedding.squeeze(0)  # (d_model,)


# For simplicity, use the [CLS] token representation (or average pooling)
# Use the first token's hidden state as the layer representation.
# layer_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
# return layer_embedding.squeeze(0) # shape: (hidden_size,)

# The aggregator model combines 32 layer embeddings into a global structure representation.

class SparseAttentionAggregator(nn.Module):
    def __init__(self, layer_embedding_dim, num_layers=32, attention_window=4, output_dim=768):
        super(SparseAttentionAggregator, self).__init__()

        self.attention = LongformerSelfAttention(hidden_size=layer_embedding_dim, attention_window=attention_window)

        self.fc = nn.Linear(layer_embedding_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, layer_embeddings):
        # (batch_size, num_layers, layer_embedding_dim)
        x = self.attention(layer_embeddings)[0]  # Apply sparse attention
        x = x.mean(dim=1)  # Mean pool across layers
        x = self.fc(x)
        return self.activation(x)