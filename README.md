


# DiffForge-XFMR: Hierarchical Structure Generation Models

This repository contains implementations of two hierarchical architectures for generating Minecraft structures. We provide two different model pipelines:

1. **Longformer-based Model:**  
   A pipeline using a Longformer-based encoder with a Masked Language Modeling objective. This setup is ideal for extracting layer-level representations from Minecraft structures.

2. **LongT5-based Model:**  
   A more advanced, conditional generation pipeline based on the LongT5 architecture. It leverages an encoder-decoder framework to reconstruct and generate complete Minecraft structures through a hierarchical approach.

Both pipelines decompose a 3D Minecraft structure (32×32×32) into 32 layers (each represented as a 1024-token sequence). The layer encoder processes each individual layer, a structure aggregator combines layer embeddings into a single global latent, and a hierarchical decoder (LongT5-based) generates or reconstructs full structures conditioned on that latent.


## Recommended Environment

We strongly recommend using a dedicated conda environment to manage dependencies and ensure reproducibility. An `environment.yml` file is provided in this repository, which includes all necessary packages.

### To Create and Activate the Environment:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/MinecraftPCG.git
   cd MinecraftPCG
   ```

2. **Create the Conda Environment:**

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Environment:**

   ```bash
   conda activate difForgeXFMR
   ```
---
---

## Model Pipelines Overview

### 1. Longformer-based Pipeline

- **Objective:**  
  Fine-tune a Longformer-based model for layer encoding using a Masked Language Modeling (MLM) objective.

- **Key Components:**
  - **Preprocessing:**  
    Raw Minecraft structures (32x32x32) are split into 32 layers, each converted into a 1024-token sequence.
  - **Layer Encoder:**  
    The model is fine-tuned on these sequences, ignoring “air” tokens, to learn robust latent representations.
  - **Aggregator & Decoder:**  
    These components can be further trained to combine layer embeddings and generate full structures.

- **Usage:**  
  Run the `encoder_training_longformer.py` script to fine-tune the layer encoder.

### 2. LongT5-based Pipeline

- **Objective:**  
  Employ a conditional generation framework using the LongT5 architecture for both layer encoding and hierarchical generation.

- **Key Components:**
  - **Preprocessing:**  
    Similar to the Longformer pipeline, structures are split into layers with special tokens (e.g., `[START_LAYER]`, `[END_ROW]`, `[END_LAYER]`).
  - **Layer Encoder:**  
    A LongT5-based encoder processes each layer and extracts high-quality embeddings.
  - **Structure Aggregator:**  
    Uses sparse attention with positional encodings to combine 32 layer embeddings into a single global latent vector.
  - **Hierarchical Decoder:**  
    A LongT5-based decoder, trained under a conditional generation objective, reconstructs complete structures conditioned on the global latent.
  - **Generation:**  
    At inference, random or interpolated global latent vectors can be sampled to generate new Minecraft structures.

- **Usage:**  
  Run the `encoder_training_longt5.py` and `decoder_training_longt5.py` (or similar) scripts to train the encoder and decoder components.


## Generating New Structures

To generate new, random structures:
1. **Sample or Interpolate a Global Latent Vector:**  
   The global latent vector is obtained from the aggregator.
2. **Feed the Latent to the Hierarchical Decoder:**  
   The decoder, conditioned on the global latent, generates 32 layer token sequences.
3. **Assemble Layers into a Full Structure:**  
   The 32 layers are combined to form a complete Minecraft structure.

This generative process leverages the conditional generation capability of LongT5 to produce coherent and diverse outputs.


## Additional Information

- **Dependencies:**  
  See the `environment.yml` file for full dependency details.
- **Documentation & References:**  
  - [LongT5: Scaling Transformers for Longer Inputs](https://arxiv.org/abs/2111.11432)
  - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
  - [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- **Usage Examples:**  
  Detailed usage examples are provided within the source code and accompanying documentation.


## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.



## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
