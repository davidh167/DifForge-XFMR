import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json, ast
import os
import numpy as np

# Import your hierarchical decoder components and helper functions.
# Ensure these are defined in your aggregator module.
from aggregator import load_layer_encoder, encode_layer
from decoder import HierarchicalDecoder

# ------------------------------
# Device Setup
# ------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# 1. Load the Pretrained Layer Encoder.
# ------------------------------
layer_encoder, layer_tokenizer = load_layer_encoder("../longformer_layer_encoder", "../longformer_layer_encoder")
layer_encoder.to(device)
layer_encoder.eval()

# ------------------------------
# 2. Load and Group the Layer Dataset.
# Each line in "layer_dataset.txt" is one layer.
# We group every 32 lines to form one full structure.
# ------------------------------
input_file = "layer_dataset.txt"
all_layers = []
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            all_layers.append(line)

num_layers_per_structure = 32
all_structures = []
for i in range(0, len(all_layers), num_layers_per_structure):
    structure = all_layers[i : i + num_layers_per_structure]
    if len(structure) == num_layers_per_structure:
        all_structures.append(structure)
        
print(f"Total structures processed: {len(all_structures)}")

# ------------------------------
# 3. Create Training Pairs.
# For each structure (32 layer strings), compute:
#   - Global latent: by encoding each layer via `encode_layer` and averaging.
#   - Target token IDs: tokenize each layer using the layer_tokenizer.
# ------------------------------
def get_global_latent(layers, layer_encoder, tokenizer, device):
    embeddings = []
    for layer in layers:
        emb = encode_layer(layer, layer_encoder, tokenizer, device)
        embeddings.append(emb)
    embeddings = torch.stack(embeddings)  # (32, hidden_size)
    global_latent = embeddings.mean(dim=0)  # (hidden_size,)
    return global_latent

training_data_file = "training_data.pt"
if os.path.exists(training_data_file):
    print(f"Loading precomputed training data from {training_data_file}...")
    training_data = torch.load(training_data_file)
    # Optionally, you could validate the loaded data (e.g., check non-zero length)
    if not training_data or len(training_data) == 0:
        print("Loaded training data is empty; recomputing training pairs.")
        compute_training_data = True
    else:
        compute_training_data = False
else:
    compute_training_data = True

if compute_training_data:
    training_data = []
    for layers in tqdm(all_structures, desc="Preparing training data"):
        global_latent = get_global_latent(layers, layer_encoder, layer_tokenizer, device)
        target_token_ids = []
        for layer in layers:
            encoding = layer_tokenizer(layer, truncation=True, padding="max_length", max_length=1014)
            target_token_ids.append(torch.tensor(encoding["input_ids"], dtype=torch.long))
        target_token_ids = torch.stack(target_token_ids)  # (32, 1014), 1014 is the max length for use with our prefix later
        training_data.append((global_latent, target_token_ids))
    torch.save(training_data, training_data_file)
    print(f"Saved training data to {training_data_file}")

print(f"Total training examples: {len(training_data)}")

# ------------------------------
# 4. Initialize the Hierarchical Decoder.
# We use a pretrained LongT5 model (via LongT5ForConditionalGeneration) as the layer decoder.
# ------------------------------
global_latent_dim = layer_encoder.config.hidden_size   # e.g., 512
layer_embedding_dim = layer_encoder.config.hidden_size   # e.g., 512
vocab_size = len(layer_tokenizer)                        # Extended vocabulary size
prompt_length = 10                                       # Length of the learned prefix for conditioning
num_layers = 32
max_length = 1024

hierarchical_decoder = HierarchicalDecoder(
    global_latent_dim, layer_embedding_dim, vocab_size, prompt_length,
    num_layers=num_layers, max_length=max_length,
    pretrained_model_name="allenai/LED-base-16384"
).to(device)

# ------------------------------
# 5. Training Loop for the Hierarchical Decoder.
# ------------------------------
optimizer = optim.Adam(hierarchical_decoder.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=layer_tokenizer.pad_token_id)
num_epochs = 3

# Initialize a GradScaler for AMP
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    total_loss = 0.0
    progress_bar = tqdm(training_data, desc=f"Epoch {epoch+1}", unit="structure")
    for global_latent, target_token_ids in progress_bar:
        # Add batch dimension (batch size = 1)
        global_latent = global_latent.unsqueeze(0).to(device)         # (1, global_latent_dim)
        target_token_ids = target_token_ids.unsqueeze(0).to(device)     # (1, 32, 1024)

        print("global_latent", global_latent.size())
        print("target_token_ids", target_token_ids.size())
        
        # For teacher forcing, you normally shift the target sequence right.
        # Here as a placeholder, we use the tokens from the first layer as the decoder input.
        decoder_input_ids = target_token_ids[:, 0, :]        # (1, 1024)
        print("decoder_input_ids", decoder_input_ids.size())
        attention_mask = torch.ones_like(decoder_input_ids)

        decoder_input_ids = decoder_input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

         # Mixed precision forward pass
        with torch.amp.autocast("cuda"):
            outputs = hierarchical_decoder(
                global_latent,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask
            )
            # outputs shape: (1, num_layers, max_length, vocab_size)
            loss = criterion(outputs.view(-1, vocab_size), target_token_ids.view(-1))
        
        # Backward pass using GradScaler for AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(training_data):.4f}")

# ------------------------------
# 6. Save the Trained Hierarchical Decoder.
# ------------------------------
torch.save(hierarchical_decoder.state_dict(), "./hierarchical_decoder.pt")
print("Saved hierarchical decoder.")