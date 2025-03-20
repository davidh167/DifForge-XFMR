"""
This script processes the tokenized structures from the input file and extracts the 32x32 layers.
Each layer is represented as a single string with special markers for the beginning and end of the layer and rows.
The output file will contain one layer per line, which can be used to train a Longformer model.

"""

import json
from tqdm import tqdm
import ast
import re

# Input file: each line is a nested list representing a 32x32x32 structure,
# but the structure is written as a single string with potential irregularities.
input_file = "tokenized_structures_new.txt"
# Output file: each line will be a flattened representation of one layer (i.e. a 32x32 grid).
output_file = "layer_dataset.txt"

all_layers = []

def extract_json_objects(text):
    """
    Extracts top-level JSON objects (delimited by matching brackets) from a string.
    This function uses a stack-based approach to capture nested arrays.
    Returns a list of substrings that each represent a JSON object.
    """
    objs = []
    stack = []
    start = None
    for i, char in enumerate(text):
        if char == '[':
            if not stack:
                start = i
            stack.append('[')
        elif char == ']':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    objs.append(text[start:i+1])
                    start = None
    return objs

def process_structure(structure):
    """
    Given a structure that is a list of 1024 rows (each row is a list of 32 tokens),
    regroup the rows into 32 layers (each layer with 32 rows).
    For each layer, insert special markers:
      - [START_LAYER] at the beginning,
      - [END_ROW] at the end of each row,
      - [END_LAYER] at the end of each layer.
    Returns a list of layer strings.
    """
    layers_flat = []
    total_rows = len(structure)       # Expected to be 1024 (32 layers x 32 rows)
    rows_per_layer = 32
    num_layers = total_rows // rows_per_layer  # Should be 32
    for i in range(num_layers):
        layer_rows = structure[i * rows_per_layer : (i + 1) * rows_per_layer]
        tokens = []
        tokens.append("[START_LAYER]")
        for row in layer_rows:
            tokens.extend(row)
            tokens.append("[END_ROW]")
        tokens.append("[END_LAYER]")
        layer_str = " ".join(tokens)
        layers_flat.append(layer_str)
    return layers_flat

# Process the input file.
with open(input_file, "r") as fin:
    for line in tqdm(fin, desc="Processing structures"):
        line = line.strip()
        if not line:
            continue
        
        # Extract all JSON objects from the line.
        json_objects = extract_json_objects(line)
        if not json_objects:
            # Fallback: try to parse the entire line as a JSON object.
            try:
                structure = json.loads(line)
                json_objects = [line]
            except Exception:
                try:
                    structure = ast.literal_eval(line)
                    json_objects = [line]
                except Exception:
                    print("Skipping malformed line.")
                    continue
        
        # Process each extracted JSON object.
        for json_obj in json_objects:
            try:
                # Replace single quotes with double quotes for JSON if necessary.
                json_obj = json_obj.replace("'", '"')
                structure = json.loads(json_obj)
            except Exception:
                try:
                    structure = ast.literal_eval(json_obj)
                except (SyntaxError, ValueError):
                    print(f"Skipping malformed JSON object: {json_obj[:100]}...")
                    continue

            # Now structure should be a nested list with shape (1024, 32).
            layers = process_structure(structure)
            all_layers.extend(layers)

print(f"Total layers extracted: {len(all_layers)}")  # Expected ~ number_of_structures * 32

# Save all layers to the output file, one layer per line.
with open(output_file, "w") as fout:
    for layer_str in all_layers:
        fout.write(layer_str + "\n")

print(f"Saved layer dataset to '{output_file}'")
