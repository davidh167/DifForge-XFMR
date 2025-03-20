import torch
import torch.nn as nn
from transformers import LongformerTokenizer
from datasets import load_dataset
import numpy as np

# Assume we have a function to load the trained layer encoder:
def load_layer_encoder(model_path, tokenizer_path):
    from transformers import LongformerForMaskedLM
    tokenizer = LongformerTokenizer.from_pretrained(tokenizer_path)
    model = LongformerForMaskedLM.from_pretrained(model_path)
    return model, tokenizer

# Function to encode a single layer using the trained layer encoder.
def encode_layer(layer_tokens, layer_encoder, tokenizer, device):
    inputs = tokenizer(layer_tokens, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = layer_encoder(**inputs, output_hidden_states=True)

    # Extract last hidden state (i.e., the last layer's output)
    hidden_states = outputs.hidden_states[-1]  # Get the last hidden layer
    
    # Extract the first token's hidden state
    layer_embedding = hidden_states[:, 0, :]  # shape: (1, hidden_size)
    return layer_embedding.squeeze(0)  # shape: (hidden_size,)


# The aggregator model combines 32 layer embeddings into a global structure representation.
class StructureAggregator(nn.Module):
    def __init__(self, layer_embedding_dim, num_layers=32, hidden_dim=256, output_dim=768):
        super(StructureAggregator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=layer_embedding_dim, nhead=4, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(layer_embedding_dim, output_dim)
    
    def forward(self, layer_embeddings):
        # layer_embeddings: (num_layers, layer_embedding_dim)
        x = layer_embeddings.unsqueeze(1)  # (num_layers, 1, d_model)
        x = self.transformer_encoder(x)      # (num_layers, 1, d_model)
        x = x.mean(dim=0)                    # (1, d_model)
        x = self.fc(x)
        return x.squeeze(0)                  # (d_model,)

