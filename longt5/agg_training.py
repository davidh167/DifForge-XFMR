import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LongT5Tokenizer
from tqdm import tqdm
from aggregator import StructureAggregator, load_layer_encoder, encode_layer  # Using Sparse Attention Aggregator

# ------------------------------
# Device Setup
# ------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Load Pretrained LongT5 Layer Encoder
# ------------------------------
layer_encoder, layer_tokenizer = load_layer_encoder("./longt5_layer_encoder", "./longt5_layer_encoder")
layer_encoder.to(device)
layer_encoder.eval()

# ------------------------------
# Load Layer Dataset
# ------------------------------
with open("layer_dataset.txt", "r") as f:
    layer_lines = [line.strip() for line in f if line.strip()]

# Group layers into full structures (each structure contains 32 layers).
num_layers_per_structure = 32
all_structure_layers = [layer_lines[i:i + num_layers_per_structure] 
                        for i in range(0, len(layer_lines), num_layers_per_structure)
                        if len(layer_lines[i:i + num_layers_per_structure]) == num_layers_per_structure]

print(f"Total structures processed: {len(all_structure_layers)}")

# ------------------------------
# Aggregator Training Function
# ------------------------------
def train_structure_aggregator(structure_layers_list, layer_encoder, tokenizer, device, num_epochs=10):
    """
    Trains the StructureAggregator to aggregate 32 layer embeddings into a single global structure latent.
    """

    aggregator = StructureAggregator(
        layer_embedding_dim=layer_encoder.config.hidden_size,  # e.g., 768 or 1024
        num_layers=32,
        attention_window=4,  # Hyperparameter for sparse attention
        output_dim=layer_encoder.config.hidden_size  # Should match global latent dimension
    ).to(device)

    optimizer = optim.Adam(aggregator.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # Loss function to approximate the global latent
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    aggregator.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(structure_layers_list, desc=f"Epoch {epoch+1}", unit="structure")

        for structure in progress_bar:
            layer_embeddings = []

            # Encode each layer using LongT5
            for layer_str in structure:
                embedding = encode_layer(layer_str, layer_encoder, tokenizer, device)
                layer_embeddings.append(embedding)

            layer_embeddings = torch.stack(layer_embeddings).to(device)  # (32, hidden_size)
            
            # Instead of using simple mean, train aggregator to produce better latent
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Enable mixed precision
                predicted_latent = aggregator(layer_embeddings)
                target = layer_embeddings.mean(dim=0)  # Baseline target, can improve later
                loss = criterion(predicted_latent, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(structure_layers_list)
        print(f"Epoch {epoch+1}, Aggregator Loss: {avg_loss:.4f}")

    return aggregator

# ------------------------------
# Train Aggregator
# ------------------------------
aggregator = train_structure_aggregator(all_structure_layers, layer_encoder, layer_tokenizer, device, num_epochs=10)

# ------------------------------
# Save Trained Aggregator
# ------------------------------
torch.save(aggregator.state_dict(), "./structure_aggregator.pt")
print("✅ Saved trained Structure Aggregator!")