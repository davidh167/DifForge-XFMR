import json, ast
import torch
import torch.nn as nn
from transformers import LongformerTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from aggregator import StructureAggregator, load_layer_encoder, encode_layer  # Assuming these are defined in aggregator.py

# Define device: if using a single GPU, this will use GPU 0.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example training loop for the aggregator.
def train_structure_aggregator(structure_layers_list, layer_encoder, tokenizer, device):
    aggregator = StructureAggregator(layer_embedding_dim=layer_encoder.config.hidden_size).to(device)
    optimizer = torch.optim.Adam(aggregator.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # Dummy target: average of layer embeddings
    
    aggregator.train()
    for epoch in range(3):
        total_loss = 0.0
        progress_bar = tqdm(structure_layers_list, desc=f"Epoch {epoch+1}", unit="structure")  # Progress bar for structures
        
        for structure in progress_bar:
            layer_embeddings = []
            for layer_str in structure:
                embedding = encode_layer(layer_str, layer_encoder, tokenizer, device)
                layer_embeddings.append(embedding)
            layer_embeddings = torch.stack(layer_embeddings)  # Shape: (32, hidden_size)
            target = layer_embeddings.mean(dim=0)  # For demonstration, using the average as target
            
            optimizer.zero_grad()
            output = aggregator(layer_embeddings)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Update tqdm bar with loss information
            progress_bar.set_postfix(loss=loss.item())
        print(f"Aggregator Epoch {epoch+1}, Loss: {total_loss/len(structure_layers_list):.4f}")
    return aggregator

# Usage:
# 1. Load the trained layer encoder.
layer_encoder, layer_tokenizer = load_layer_encoder("./longformer_layer_encoder", "./longformer_layer_encoder")
layer_encoder.to(device)
layer_encoder.eval()

# 2. Load your preprocessed layer dataset.
# Here, we assume that "layer_dataset.txt" was generated so that each line is one flattened layer.
with open("layer_dataset.txt", "r") as f:
    layer_lines = [line.strip() for line in f if line.strip()]

# Now, group the layers into full structures.
# For example, if the original dataset had 2197 structures and each structure
# was split into 32 layers, then layer_lines should have 2197*32 lines.
num_layers_per_structure = 32
all_structure_layers = []
for i in range(0, len(layer_lines), num_layers_per_structure):
    structure_layers = layer_lines[i:i + num_layers_per_structure]
    if len(structure_layers) == num_layers_per_structure:
        all_structure_layers.append(structure_layers)

print(f"Total structures processed: {len(all_structure_layers)}")

# 3. Train the aggregator using the grouped layer data.
aggregator = train_structure_aggregator(all_structure_layers, layer_encoder, layer_tokenizer, device)

# Save the aggregator model.
torch.save(aggregator.state_dict(), "./structure_aggregator.pt")
print("Saved structure aggregator.")
